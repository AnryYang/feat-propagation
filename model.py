#########################################################################
# File Name: model.py
# Author: supergx
# mail: yang0461@ntu.edu.sg
# Created Time: Fri 15 May 2020 10:09:10 AM
#########################################################################
#!/usr/bin/env/ python

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, JumpingKnowledge, GraphConv, ChebConv, SAGEConv, GATConv
from torch_geometric.data import Data
from scipy.sparse import csr_matrix, coo_matrix, hstack, identity, csgraph, lil_matrix
from sklearn import preprocessing
import time
from sklearn.metrics import *
import random
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis, KernelPCA, NMF
from torch_geometric.utils import *

from sklearn.utils.extmath import randomized_svd

SEED=1234
def fix_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=True

fix_seed()

class Param():
    def __init__(self, model_type=1, num_epoch=400, num_layers=2, hidden=200, dropout=0.5, lr=0.005):
        self.model_type = model_type
        self.num_epoch = num_epoch
        self.num_layers = num_layers
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.improved = False
    
    def display(self):
        print("model-type=%d, epoch=%d, layers=%d, hidden=%d, dropout=%f, lr=%f, improved=%d"%(self.model_type, self.num_epoch, self.num_layers, self.hidden, self.dropout, self.lr, self.improved))


def propagate_by_lapacian(W, x, t):
    W = csgraph.laplacian(W, normed=True)
    W.data = W.data*-1
    w = x
    for i in range(t):
        print("%d-th iteration"%(i+1))
        w = W.dot(w)+x

    x = w
    x.data *= x.data>0
    x.eliminate_zeros()
    return x

def propagate_by_ppr(W, x, t):
    W = preprocessing.normalize(W, norm='l1', axis=1)
    W = preprocessing.normalize(W, norm='l1', axis=0)
    W = W.T
    w = x
    for i in range(t):
        print("%d-th iteration"%(i+1))
        w = W.dot(w)+x
    del W
    x = w
    return x

class LIN(torch.nn.Module):
    def __init__(self, num_layers=2, hidden=200, features_num=16, num_class=2, dropout=0.5):
        super(LIN, self).__init__()
        self.lin2 = Linear(hidden, num_class)
        self.first_lin = Linear(features_num, hidden)
        self.dropout=dropout

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
        

class GGCN(torch.nn.Module):
    def __init__(self, num_layers=2, hidden=200, features_num=16, num_class=2, dropout=0.5):
        super(GGCN, self).__init__()
        self.conv1 = GraphConv(features_num, hidden, aggr='add')
        self.lin2 = Linear(hidden, num_class)
        self.dropout = dropout
        print("hidden=%d, dropout=%f"%(hidden, self.dropout))

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class Model:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def generate_pyg_data(self, data):
        fix_seed()
        x = data['fea_table']
        if x.shape[1] == 1:
            x = x.to_numpy()
            x = x.reshape(x.shape[0])
            x = np.array(pd.get_dummies(x))
        else:
            x = x.drop('node_index', axis=1).to_numpy()
        
        num_nodes = x.shape[0]
        num_feats = x.shape[1]

        df = data['edge_file']
        edge_index = df[['src_idx', 'dst_idx']].to_numpy()
        edge_index = sorted(edge_index, key=lambda d: d[0])
        edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)

        edge_weight = df['edge_weight'].to_numpy()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)

        # adding self-loops is good for dataset 1 and 4, but not for dataset 3 and 5
        avg_degree = edge_weight.numel()*1.0/num_nodes
        if avg_degree<100:  # the avg degree is too low, add self-loops to enrich
            edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight=edge_weight)
            print("self-loop: ", contains_self_loops(edge_index=edge_index))
            print(edge_index.size(), edge_weight.size())
             
        x[x<0]=0
        x = csr_matrix(x)
        print("n=%d, d=%d, numer-of-feats=%d, density=%f"%(num_nodes, num_feats, x.getnnz(), x.getnnz()*1.0/num_nodes/num_feats))
        
        #W = csr_matrix((edge_weight, (df['src_idx'].to_numpy(), df['dst_idx'].to_numpy())), shape=(num_nodes, num_nodes))
        W = csr_matrix((edge_weight, (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
        print("m=%d, n=%d, avg-degree=%f, density=%f"%(W.getnnz(), W.shape[0], W.getnnz()*1.0/W.shape[0], W.getnnz()*1.0/W.shape[0]/W.shape[0]))
        
        if is_undirected(edge_index)==False:
            print("directed: True")
            W = W+W.T

        avg_degree = edge_weight.numel()*1.0/num_nodes
        
        y = torch.zeros(num_nodes, dtype=torch.long)
        inds = data['train_label'][['node_index']].to_numpy()
        train_y = data['train_label'][['label']].to_numpy()
        y[inds] = torch.tensor(train_y, dtype=torch.long)
        labels = set(train_y.flatten())
        print(labels)
        num_label = len(labels)
        x2 = csr_matrix(([1]*inds.shape[0], (inds.flatten(), train_y.flatten())),shape=(num_nodes, num_label))
            
        param = Param()
            
        if x.getnnz()<=num_nodes: # without input features, homogeneous graphs
            print("homogeneous graph")
            avg_density = W.getnnz()*1.0/W.shape[0]/W.shape[0]
            t = min(5,int(np.log(1.0/avg_density)))
            x = W
            if avg_degree<3.5:
                print("flag is on 1")
                x = propagate_by_lapacian(W, x, t)
                x2 = propagate_by_lapacian(W, x2, t)
                param.hidden=300
            else:
                print("flag is on 3")
                if avg_degree>100:
                    print("svd...")
                    k = min(max(800,int(num_nodes/12)),num_nodes)
                    svd = TruncatedSVD(n_components=k, random_state=SEED)
                    x = svd.fit_transform(x)
                    x = csr_matrix(x)
                x = propagate_by_ppr(W, x, t)
                x2 = propagate_by_ppr(W, x2, t)
                param.hidden=200
            x = hstack([x,x2])
            param.model_type=1
            param.dropout=0.5
        else: # with input features, attributed graphs
            print("attributed graph")
            if num_feats>2000 or avg_degree>100: # dataset 2. rich features or high degree -> expensive propagation costs
                print("flag is on 4")
                t = 0
                param.hidden=200
            else:  # dataset 1. poor features and not high degree -> inexpensive propagation costs
                print("flag is on 5")
                t = 2
                param.hidden=300
            xtmp = hstack([W,x2])
            xtmp = propagate_by_ppr(W, xtmp, t)
            xxtmp = x
            xxtmp = propagate_by_ppr(W, xxtmp, t+1)
            x = hstack([xxtmp,xtmp])
            x = preprocessing.normalize(x, norm='l2', axis=1)
            param.model_type=0
            param.lr = 0.01
            param.dropout = 0.3

        x_density = x.getnnz()*1.0/(x.shape[0]*x.shape[1])
        print("nnz-x=%d, x-density=%f"%(x.getnnz(), x_density))
        
        x = x.todense()
        
        x = torch.tensor(x, dtype=torch.float)
        print(x.size())
        num_nodes = x.size(0)

        train_indices = data['train_indices']
        test_indices = data['test_indices']

        data = Data(x=x, edge_index=edge_index, y=y, edge_weight=edge_weight)

        data.num_nodes = num_nodes

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_indices] = 1
        data.train_mask = train_mask

        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[test_indices] = 1
        data.test_mask = test_mask
        return data, param, train_y

    def train(self, data, param, train_y):
        fix_seed()
        if param.model_type==1:
            print("training with GGCN model...")
            model = GGCN(num_layers=param.num_layers, features_num=data.x.size()[1], num_class=int(max(data.y)) + 1, hidden=param.hidden, dropout=param.dropout)
        else:
            print("training with linear model...")
            model = LIN(num_layers=param.num_layers, features_num=data.x.size()[1], num_class=int(max(data.y)) + 1, hidden=param.hidden, dropout=param.dropout)
        
        model = model.to(self.device)
        data = data.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=param.lr, weight_decay=5e-4)

        min_loss = float('inf')
        for epoch in range(1,param.num_epoch):
            model.train()
            optimizer.zero_grad()
            loss = F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if epoch%20==0:
                torch.cuda.empty_cache()
                model.eval()
                with torch.no_grad():  
                    train_pred = model(data)[data.train_mask].max(1)[1].cpu().numpy().flatten()
                    
                acc = accuracy_score(train_y, train_pred)
                print("Accuracy@%d epoch: %f, loss=%f"%(epoch, acc, loss.item()))

        return model

    def pred(self, model, data):
        fix_seed()
        model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            pred = model(data)[data.test_mask].max(1)[1]
        return pred

    def train_predict(self, data, time_budget,n_class,schema):
        t0 = time.time()
        data, param, train_y = self.generate_pyg_data(data)
        param.display()
        t1 = time.time()
        print("preprocessing time: %f"%(t1-t0))
        model = self.train(data, param, train_y)
        pred = self.pred(model, data)

        return pred.cpu().numpy().flatten()
