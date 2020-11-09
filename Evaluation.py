import numpy as np
import math
from scipy.special import comb

def evaluation(data,result,label):#注输入向量均为numpy.array类型
    '''此函数主要是计算各种评价指标
    data:图的邻接矩阵 data[i,j]=1表示第i个节点与第j个节点之间存在边的连接 data[i,j]=0表示第i个节点与第j个节点之间不存在边的连接
    result:表示聚类的结果,其是一个行向量1*n,每一列代表一个样本的类标签,np.array([[]])
    label:表示数据的真实类标签,其是一个行向量1*n,每一列代表一个样本的类标签
    '''
    Q=0#图的模块度
    m=0#计算图边的个数
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i,j]>0:
                m = m +1
    m = m/2
    #计算模块度，Newman M E J. Communities, modules and large-scale structure in networks[J]. Nature physics, 2012, 8(1): 25.
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if result[0,i] == result[0,j]:#两个节点处于同一社团中
                if i!=j:#两个节点不是同一个节点
                   Q = Q + 1/(2*m)*(data[i,j] - 1/(2*m)*sum(data[i,:])*sum(data[j,:]))
    #主要计算各种度量指标
    if len(label)!=0:#label为不为空
        TP=0#初始化
        FP=0
        FN=0
        TN=0
        for i in range(result.shape[1]-1):
            for j in range(i+1,result.shape[1]):
                if result[0,i]==result[0,j]:#两个样本聚在同一个类簇中
                    if label[0,i]==label[0,j]:#两个样本在实际也处于同一类簇中
                        TP=TP+1;
                    else:#两个样本在实际不处于同一类簇中
                        FP=FP+1
                else:#两个样本聚不在同一个类簇中
                    if label[0,i]==label[0,j]:#两个样本在实际处于同一类簇中
                        FN=FN+1;
                    else:#两个样本在实际不处于同一类簇中
                        TN=TN+1
        FM=0
        K=0
        RT=0
        recall=0
        F1=0
        J = 0
        J = TP/(TP+FP+FN+TN)
        FM=TP/math.sqrt((TP+FN)*(TP+FP))#计算的是Folkes and Mallows指标[0,1],值越大则结果越好
        K=0.5*(TP/(TP+FP)+TP/(TP+FN))#计算的是Kulczynski指标[0,1],值越大越好,0表示完全不一致,1表示完全一致
        RT=(TP+TN)/(TP+TN+2*(FN+FP))#计算的是Rogers-Tanimoto指标[0,1]
        precision=TP/(TP+FP)#计算的是precision指标
        recall=TP/(TP+FN)#计算的是召回率指标
        F1=(2*precision*recall)/(precision+recall)#计算的是F-measure指标
        return J,FM,K,recall,F1#返回计算的结果
    else:
        return 0,0,0,0,0
