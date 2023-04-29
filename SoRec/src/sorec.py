import math
import numpy as np
import time
import copy
import math

from numpy.random import RandomState
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from matplotlib import pyplot as plt
from numpy import random
from tqdm import tqdm

def fx(x):
    return (x - 1) / 4

def logistic(x):
    return 1 / (1 + np.exp(-x))

# The derivative of logistic
def logistic_deri(x):
    val = logistic(x)
    return val*(1-val)

class SoRec_model:
    def __init__(self, R, Validate, C, lr = 0.01, lambda_c=0.1, lambda_u=0.001, lambda_v=0.001, lambda_z=0.001, dim=10,iters=1000,seed=None):
        self.R = R
        self.Validate= Validate
        self.C = C
        self.lambda_c = lambda_c
        self.lambda_u = lambda_u
        self.lambda_z = lambda_z
        self.lambda_v = lambda_v
        self.dim = dim
        self.random_state = RandomState(seed)
        self.iters = iters
        self.lr = lr
        self.U = np.mat(self.random_state.rand(dim, np.size(R, 0)))
        self.V = np.mat(self.random_state.rand(dim, np.size(R, 1)))
        self.Z = np.mat(self.random_state.rand(dim, np.size(C, 1)))

    # the MAE for train set
    def train_loss(self, UVdata):
        loss = (np.fabs(4 * logistic(UVdata) + 1 - self.R.data)).sum()
        loss /= self.R.shape[0]
        return loss

    # the MAE for validate_set
    def evaluation(self):
        mae = 0.0
        index = self.Validate.nonzero()
        data = self.Validate.data
        total = data.shape[0]
        for i in range(total):
            predict = 4*logistic((self.U[:,index[0][i]].T*self.V[:,index[1][i]])[0,0])+1
            mae += math.fabs(data[i] - predict)
        return mae / total


    def train(self):
        print("Train start")
        Rindex = self.R.nonzero()
        Cindex = self.C.nonzero()
        
        Rdata = fx(self.R.data)
        Cdata = self.C.data
        
        Rnum = Rdata.shape[0]
        Cnum = Cdata.shape[0]
        
        UVdata = copy.deepcopy(Rdata)
        UZdata = copy.deepcopy(Cdata)
        
        train_loss_list = []
        validate_loss_list = []
        begin = time.time()
        for it in tqdm(range(self.iters)):
            start = time.time()
            for k in range(Rnum):
                UVdata[k] = (self.U[:, Rindex[0][k]].T * self.V[:, Rindex[1][k]])[0][0]
            for k in range(Cnum):
                UZdata[k] = (self.U[:, Cindex[0][k]].T * self.Z[:, Cindex[1][k]])[0][0]

            UV = csr_matrix(((logistic_deri(UVdata) * logistic(UVdata) - Rdata), Rindex), self.R.shape)
            UZ = csr_matrix(((logistic_deri(UZdata) * logistic(UZdata) - Cdata), Cindex), self.C.shape)

            U = csr_matrix(self.U)
            V = csr_matrix(self.V)
            Z = csr_matrix(self.Z)

            grads_u = self.lambda_u * U + UV.dot(V.T).T + self.lambda_c * UZ.dot(Z.T).T
            grads_v = UV.T.dot(U.T).T + self.lambda_v * V
            grads_z = self.lambda_c * UZ.T.dot(U.T).T + self.lambda_z * Z

            self.U = self.U - self.lr*grads_u
            self.V = self.V - self.lr*grads_v
            self.Z = self.Z - self.lr*grads_z

            trloss = self.train_loss(UVdata)
            valiloss = self.evaluation()
            train_loss_list.append(trloss)
            validate_loss_list.append(valiloss)
            end = time.time()
            print("iter:{}, last_train_loss:{}, validate_loss:{}, timecost:{}, have run:{}".format(it + 1, trloss, valiloss, end - start, end - begin))

        x = np.linspace(1, self.iters, self.iters)
        plt.plot(x, train_loss_list, label='train_loss')
        plt.show()
        plt.plot(x, validate_loss_list, label='validate_loss')
        plt.show()
        return self.U, self.V, self.Z, train_loss_list, validate_loss_list

def get_trust_data(filename="epinions_dataset/trust_data.txt",theshape=(49290,49290)):
    f = open("../epinions_dataset/trust_data.txt")
    lines = f.readlines()
    row = []
    col = []
    data = []
    for line in tqdm(lines):
        alist = line.strip('\n').split()
        row.append(int(alist[0])-1)
        col.append(int(alist[1])-1)
        data.append(float(alist[2]))
    mtx = coo_matrix((data, (row, col)), shape=theshape)
    indeg = mtx.sum(axis=0)
    outdeg = mtx.sum(axis=1)
    factor = copy.deepcopy(mtx)
    for k in range(factor.data.shape[0]):
        i = factor.row[k]
        j = factor.col[k]
        factor.data[k] = math.sqrt(indeg[0, j]/(indeg[0,j]+outdeg[i, 0]))
    return csr_matrix(factor)

def get_ratings_data(filename="epinions_dataset/ratings_data.txt",theshape=(49290,139739)):
    with open("../epinions_dataset/ratings_data.txt") as f:
        lines = f.readlines()

    train_data = []
    train_row = []
    train_col = []
    vali_data = []
    vali_row = []
    vali_col = []

    random.shuffle(lines)
    ind = -1
    pos = int(len(lines)*0.99)
    for line in tqdm(lines):
        ind += 1
        alist = line.strip('\n').split()
        if ind>=pos:
            vali_row.append(int(alist[0]))
            vali_col.append(int(alist[1]))
            vali_data.append(int(alist[2]))
            continue
        train_row.append(int(alist[0]))
        train_col.append(int(alist[1]))
        train_data.append(int(alist[2]))

    train_mtx = csr_matrix((train_data, (train_row,train_col)), shape=theshape, dtype='float64')
    vali_mtx = csr_matrix((vali_data, (vali_row, vali_col)), shape=theshape, dtype='float64')
    return train_mtx, vali_mtx

if __name__ == '__main__':
    trust_data = get_trust_data()
    ratings_data_train, ratings_data_validate = get_ratings_data()
    socmodel = SoRec_model(ratings_data_train, ratings_data_validate, trust_data, lr=0.01, dim=10,iters=100)
    U,V,Z,train_loss_list, validate_loss_list = socmodel.train()