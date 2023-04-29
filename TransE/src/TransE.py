import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import math

import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import DataLoader
from torch.utils import data
from tqdm import tqdm
from torch.optim import AdamW
import torch.optim as optim
from torch.nn.functional import normalize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# from datasets import Dataset
def hit_at_k(predictions: torch.Tensor, ground_truth_idx: torch.Tensor, k: int = 10) -> int:
    assert predictions.size(0) == ground_truth_idx.size(0)

    zero_tensor = torch.tensor([0]).to(device)
    one_tensor = torch.tensor([1]).to(device)
    _, indices = predictions.topk(k=k, largest=False)
    return torch.where(indices == ground_truth_idx, one_tensor, zero_tensor).sum().item()

# def mean_rank(predictions, ground_truth_idx):
#     rank = 0
#     for i in range(predictions.shape[0]):
#         index = torch.squeeze(ground_truth_idx)[i] - 1
#         rank += (predictions[i] < predictions[i][index]).sum() + (predictions[i] == predictions[i][index]).sum()
#     return rank

def create_mappings(dataset_path):
    entity_counter = Counter()
    relation_counter = Counter()
    with open(dataset_path, "r") as f:
        for line in f:
            head, relation, tail = line[:-1].split("\t")
            entity_counter.update([head, tail])
            relation_counter.update([relation])
            
    entity2id = {}
    relation2id = {}
    for idx, (mid, _) in enumerate(entity_counter.most_common()):
        entity2id[mid] = idx
    for idx, (relation, _) in enumerate(relation_counter.most_common()):
        relation2id[relation] = idx
    return entity2id, relation2id

class customDataset(data.Dataset):
    def __init__(self, fb15k_path, entity2id, relation2id):
        self.entity2id = entity2id
        self.relation2id = relation2id
        with open(fb15k_path, "r") as f:
            self.data = [line[:-1].split("\t") for line in f]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        head, relation, tail = self.data[index]
        head_id = self._to_idx(head, self.entity2id)
        relation_id = self._to_idx(relation, self.relation2id)
        tail_id = self._to_idx(tail, self.entity2id)
        return head_id, relation_id, tail_id  

    def _to_idx(self, key, mapping):
        try:
            return mapping[key]
        except KeyError:
            return len(mapping)


def load_data(fb15k_path, batch_size, entity2id, relation2id):
    dataset = customDataset(fb15k_path, entity2id, relation2id)
    data_loader = DataLoader(dataset, batch_size = batch_size)
    return data_loader
    
class trans_E(nn.Module):
    def __init__(self, entity_num, label_num, margin, embed_dim):
        super().__init__()
        
        self.margin = margin
        self.embed_label = nn.Embedding(num_embeddings=label_num + 1,
                                            embedding_dim = embed_dim)
        self.embed_entity = nn.Embedding(num_embeddings=entity_num + 1,
                                            embedding_dim = embed_dim)
        
        bound = 6  / np.sqrt(embed_dim)
        nn.init.uniform_(self.embed_entity.weight, -bound, bound)
        nn.init.uniform_(self.embed_label.weight, -bound, bound)
        
        self.embed_label.weight.data[:-1, :].div_(self.embed_label.weight.data[:-1, :].norm(p=2, dim=1, keepdim=True))
      
    def distance(self, triplet):
        heads = triplet[:, 0]
        labels = triplet[:, 1]
        tails = triplet[:, 2]
        distance = torch.norm(self.embed_entity(heads) + self.embed_label(labels) - self.embed_entity(tails), 2, dim=1)
        return (distance)
    
    def predict(self, triplets):
        return self.distance(triplets)
        
    def forward(self, positive, negative):
        self.embed_entity.weight.data[:-1, :].div_(self.embed_entity.weight.data[:-1, :].norm(p=2, dim=1, keepdim=True))
        positive = self.distance(positive)
        negative = self.distance(negative)
        loss = torch.max(torch.zeros(1,1).to(device), self.margin + positive - negative)
        return loss

def train(model, epochs, train_data, valid_data, learning_rate, entity2id, relation2id):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_list = []
    valid_list = []
    model.eval()
    valid = evaluation(model, valid_data, len(entity2id))
    print("init validation is : ", valid)
    valid_list.append(valid)

    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        for head, label, tail in train_data:
            model.zero_grad()
            head = head.to(device)
            label = label.to(device)
            tail = tail.to(device)

            positive_triples = torch.stack((head, label, tail), dim=1).to(device)
            head_or_tail = torch.randint(high=2, size=head.size()).to(device)
            random_entities = torch.randint(high=len(entity2id), size=head.size()).to(device)
            broken_heads = torch.where(head_or_tail == 1, random_entities, head).to(device)
            broken_tails = torch.where(head_or_tail == 0, random_entities, tail).to(device)
            negative_triples = torch.stack((broken_heads, label, broken_tails), dim=1).to(device)
            
            loss = model(positive_triples, negative_triples)
            mean_loss = loss.mean()
            total_loss += mean_loss
            loss.mean().backward()
            optimizer.step()
            
        loss_list.append((total_loss/len(train_data)).data.cpu())
        model.eval()
        if (epoch+1)%20 == 0:
            valid = evaluation(model, valid_data, len(entity2id))
            print("validation is : ", valid)
            valid_list.append(valid)
    
    print("training ended")
    plt.plot(loss_list)
    plt.show()
    plt.plot(valid_list)
    plt.show()
    print(mean_loss)
        
    

def evaluation(model, test_data, entities_count):
    model.eval()
    examples_count = 0.0
    hits_at_10 = 0.0
    
    entity_ids = torch.arange(end=entities_count).unsqueeze(0)
    print("evaluation start \n")
    for head, relation, tail in tqdm(test_data):
        current_batch_size = head.size()[0]
        all_entities = entity_ids.repeat(current_batch_size, 1)
        heads = head.reshape(-1, 1).repeat(1, all_entities.size()[1])
        relations = relation.reshape(-1, 1).repeat(1, all_entities.size()[1])
        tails = tail.reshape(-1, 1).repeat(1, all_entities.size()[1])

        # Check all possible tails
        triplets = torch.stack((heads, relations, all_entities), dim=2).reshape(-1, 3).to(device)
        tails_predictions = model.predict(triplets).reshape(current_batch_size, -1).to(device)

        # Check all possible heads
        triplets = torch.stack((all_entities, relations, tails), dim=2).reshape(-1, 3).to(device)
        heads_predictions = model.predict(triplets).reshape(current_batch_size, -1).to(device)

        # Concat predictions
        predictions = torch.cat((tails_predictions, heads_predictions), dim=0)
        ground_truth_entity_id = torch.cat((tail.reshape(-1, 1), head.reshape(-1, 1))).to(device)
        hits_at_10 += hit_at_k(predictions, ground_truth_entity_id, k=10)
        examples_count += predictions.size()[0]

    hits_at_10_score = hits_at_10 / examples_count * 100
    return hits_at_10_score


def main():
    # for fb15k embed_dim = 50, lr = 0.01, margin = 1, and d = L1
    margin = 1
    lr = 0.01
    embed_dim = 50
    batch_size = 512
    epochs = 100
    
    entity2id, relation2id = create_mappings('../data/train.txt')
    train_loader = load_data('../data/train.txt', batch_size, entity2id, relation2id)
    valid_loader = load_data('../data/valid.txt', batch_size, entity2id, relation2id)
    test_loader = load_data('../data/test.txt', batch_size, entity2id, relation2id)
    model = trans_E(len(entity2id), len(relation2id), margin, embed_dim)
    model.to(device)
    train(model, epochs, train_loader, valid_loader, lr, entity2id, relation2id)
    hit = evaluation(model, test_loader, len(entity2id))
    print("test set 10@hits is ", hit)

main()