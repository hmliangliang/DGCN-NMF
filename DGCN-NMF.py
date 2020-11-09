import numpy as np
import torch
import dgl
import torch.nn as nn
from torch.nn import functional as F
import networkx as nx
from dgl.nn.pytorch import GraphConv
import time
from sklearn.decomposition import non_negative_factorization
from torch.nn import BatchNorm1d
import torch.optim as optim
from Evaluation import evaluation
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

class Net(nn.Module):
    def __init__(self,parameters,graph):
        super(Net,self).__init__()
        self.Parameters = parameters
        self.graph = dgl.DGLGraph(graph) #graph must be the type of DGL graph
        self.cov1 = GraphConv(self.graph.number_of_nodes(), self.Parameters.layers[0])
        self.cov2 = GraphConv(self.Parameters.layers[0]+self.Parameters.d, self.Parameters.layers[1])
        self.f = nn.Linear(self.Parameters.layers[1],self.Parameters.layers[2])
        self.f2 = nn.Linear(self.Parameters.layers[2], self.Parameters.k)
    def forward(self, X):
        u0, _, _ = non_negative_factorization(np.array(nx.adjacency_matrix(self.graph.to_networkx()).todense()),n_components=self.Parameters.d)#The adjacency matrix of graph is decomposed
        u0 = torch.tensor(u0, dtype=torch.float)
        model = BatchNorm1d(X.shape[1],affine=True)
        X = model(X)
        h = self.cov1(self.graph, X)
        if self.Parameters.fun == 'sigmoid':
            h = F.sigmoid(h)
        elif self.Parameters.fun == 'ReLU':
            h = F.relu(h)
        u1, _, _ = non_negative_factorization(h.detach().numpy(),n_components=self.Parameters.d)
        u1 = torch.tensor(u1,dtype=torch.float)
        h = torch.cat((h, u0), dim=1)
        model = BatchNorm1d(h.shape[1], affine=True)
        h = model(h)
        h = self.cov2(self.graph, h)
        if self.Parameters.fun == 'sigmoid':
            h = F.sigmoid(h)
        elif self.Parameters.fun == 'ReLU':
            h = F.relu(h)
        if self.Parameters.attribute == True:
            h = torch.cat((h,self.Parameters.X), dim=1)
        model = BatchNorm1d(h.shape[1], affine=True)
        h = model(h)
        h = self.f(h)
        h = F.sigmoid(h)
        h = self.f2(h)
        return h
    def loss(self,h):
        lr = 0
        lm = 0
        ha = torch.argmax(h, dim=1) #The assignment of community discovry
        n = h.shape[0]
        lr = lr + torch.norm(torch.mm(h, h.transpose(0,1)) - torch.tensor(np.array(nx.adjacency_matrix(self.graph.to_networkx()).todense()),dtype=torch.float),'fro')
        g = self.graph.to_networkx()
        adj = nx.edges(g)
        m = self.graph.num_edges()
        A = np.array(nx.adjacency_matrix(self.graph.to_networkx()).todense())
        for i in range(n):# COmpute modularity
            for j in range(n):
                if A[i,j] > 0:
                    if ha[i]  == ha[j]:
                        k1 = sum(A[i,:])
                        k2 = sum(A[j,:])
                        lm = lm + 1 / (2 * m) * (1 - k1 * k2 / (2 * m))
                else:
                    if ha[i]  == ha[j]:
                        k1 = sum(A[i,:])
                        k2 = sum(A[j,:])
                        lm = lm + 1 / (2 * m) * (0 - k1 * k2 / (2 * m))
        lm = torch.tensor(lm,dtype=torch.float)
        #print('lr=',lr, '  lm=',lm)
        return lr - self.Parameters.alpha*lm


class Parameters():
    def __init__(self, X, attribute = False, k=2, fun = 'ReLU',d=64, layers=[150,100,64], alpha=1):
        self.attribute = attribute #attribute = True if graph is an attributed graph; otherwise, attribute =False
        self.k = k #The number of communities
        self.alpha = alpha
        self.d = d #The embedding dimension of NMF
        self.layers = layers #THe output dimension of layers
        self.fun = fun #fun = 'ReLU' or 'sigmoid' is the activation funtion
        if self.attribute == False:
            self.X = None
        else:
            self.X = X# THe attribute features of nodes

if __name__ == '__main__':
    start = time.time()
    data = np.loadtxt('./data/adjnoun/adjnoun-graph.txt',dtype=int).tolist() #adjnoun/adjnoun
    labels = np.loadtxt('./data/adjnoun/adjnoun-labels.txt',dtype=int)[:,1]
    graph = nx.Graph()
    k = len(np.unique(labels))
    graph.add_nodes_from([i for i in range(len(labels))])
    graph.add_edges_from(data)
    epoch_max = 200
    input_fg = int(input('Is graph an attributed graph? (1.Yes. 2. No) 1 or 2:'))
    if input_fg == 1:
        input_fg = True
        X = torch.tensor(np.loadtxt('../data/adjnoun/adjnoun.txt'), dtype=torch.float)
    else:
        input_fg = False
        X = torch.eye(graph.number_of_nodes())
    parameters = Parameters(X = X, attribute = input_fg, k=k, fun = 'ReLU',d=64, layers=[150,100,64], alpha=20)
    net = Net(parameters, graph)
    optimizer = optim.Adam(net.parameters(), lr=0.0005, betas = (0.9,0.99))
    Loss_list = []
    for epoch in range(epoch_max):
        optimizer.zero_grad()
        h = net.forward(X)
        loss = net.loss(h)
        loss.backward()
        optimizer.step()
        Loss_list.append(loss.item())
        print('The',epoch,'th training process, loss=',round(loss.item(),4))
    f = open('DGCN-NMF-ReLU-loss-adjnoun(alpha=20).txt','w')
    f.write(str(Loss_list))
    f.close()
    y_pred = torch.argmax(h,dim=1).numpy()
    J, FM, K, recall, F1 = evaluation(np.array(nx.adjacency_matrix(graph).todense()),np.array([y_pred]), np.array([labels]))
    print('J=',round(J,4))
    print('FM=', round(FM, 4)
    print('K=', round(K, 4))
    print('recall=', round(recall, 4))
    print('F1=', round(F1, 4))
    end = time.time()
    print('Time cost =', round(end-start, 4))
