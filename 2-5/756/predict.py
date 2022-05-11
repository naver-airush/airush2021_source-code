import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch.nn as nn
from sklearn.cluster import KMeans
# from kmeans_pytorch import kmeans
from sklearn.decomposition import PCA
# import faiss
import time

def predict_infer(args, model, dataloader):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    label = []
    result = []
    
    for i, (nv_mid ,prod_nm) in enumerate(dataloader):
        
        nv_mid = list(nv_mid)
        prod_nm = list(prod_nm)
        with torch.no_grad():
            logits = model.encode(prod_nm)
            if i == 0:
                result = logits
            else:
                result = np.concatenate((result,logits),axis=0) 
        label = label + nv_mid

    return result, label


def predict(args, model, dataloader):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    label = []
    result = []

    for i, (nv_mid ,prod_nm, match_nv_mid) in enumerate(dataloader):
        
        nv_mid = list(nv_mid)
        prod_nm = list(prod_nm)
        match_nv_mid = list(match_nv_mid)

        with torch.no_grad():
            logits = model.encode(prod_nm)
        if i == 0:
            result = logits
        else:
            result = np.concatenate((result,logits),axis=0) 
        label = label + nv_mid

    return result, label



        # return [('1906368762', ['1906368762','1810466025','5159532445']),
        #         ('636762', ['636762','1146025','155245'])] 

def result_maker(data_result,data_label,query_result,query_label):
    
    # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    data_result = torch.tensor(data_result).to('cuda')
    query_result = torch.tensor(query_result).to('cuda')
    result_list = []

    for query in query_label:
        result_list.append((query,[])) ## [('1906368762', ['1906368762','1810466025','5159532445']], ['636762', ['636762','1146025','155245']]]

    for i, data in enumerate(data_result):
        # print('{}번째'.format(i),'data간의 거리차 구하는 중..' ,'/', '총 {} 개'.format(len(data_label)))
        sims = util.pytorch_cos_sim(data.unsqueeze(dim=0),query_result)[0]
        sims = sims.cpu()
        max_index = torch.argmax(sims)

        if sims[max_index] < 0.652:
            continue
        else:
            result_list[max_index][1].append(data_label[i])
        # result_list[max_index][1].append(data_label[i])

    return result_list

def result_maker_for_test(data_result,data_nv_mid,query_result,query_nv_mid,threshold=None):
    
    # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    data_result = torch.tensor(data_result).to('cuda')
    query_result = torch.tensor(query_result).to('cuda')

    result_list = []
    for nv_mid in query_nv_mid:
        result_list.append((nv_mid,[])) ## [('1906368762', ['1906368762','1810466025','5159532445']], ['636762', ['636762','1146025','155245']]]

    for i, data in enumerate(data_result):
        # print('{}번째'.format(i),'data간의 거리차 구하는 중..' ,'/', '총 {} 개'.format(len(data_label)))
        sims = util.pytorch_cos_sim(data.unsqueeze(dim=0),query_result)[0]
        sims = sims.cpu()
        max_index = torch.argmax(sims)

        if sims[max_index] < threshold:
            continue
        else:
            result_list[max_index][1].append(data_nv_mid[i])
        # result_list[max_index][1].append(data_label[i])

    return result_list

def l2_result_maker(data_result,data_label,query_result,query_label):
    data_result = torch.tensor(data_result).to('cuda')
    query_result = torch.tensor(query_result).to('cuda')
    result_list = []

    for query in query_label:
        result_list.append((query,[])) ## [('1906368762', ['1906368762','1810466025','5159532445']], ['636762', ['636762','1146025','155245']]]

    for i, data in enumerate(data_result):
        # print('{}번째'.format(i),'data간의 거리차 구하는 중..' ,'/', '총 {} 개'.format(len(data_label)))
        minus = data.unsqueeze(dim=0)-query_result
        square = minus*minus
        sum = torch.sum(square,dim=1)
        max_index = torch.argmin(sum)
        softmax_result = nn.Softmax(-sum)
        softmax_result = nn.Softmax(-sum)
        result_list[max_index][1].append(data_label[i])

    return result_list

def k_mean_result_maker(data_result,data_label,query_result,query_label):
    print('clustering 시작..')

    ##PCA##
    # pca = PCA(n_components=10)
    # data_result = pca.fit_transform(data_result)
    ##PCA##
    start = time.time()
    print('input:', data_result.shape)
    print('kmean 시작')    
    kmean = KMeans(n_clusters=len(query_label),query_result=query_result)
    # kmean = FaissKMeans(n_clusters=10)
    kmean.fit(data_result)
    print('kmean 끝')
    print(round(time.time()-start,2),"초")
    cluster_assignment = kmean.predict(data_result).transpose().squeeze()

    data_label = np.array(data_label)
    

    result_list = []
    
    for i, query in enumerate(query_label):
        print('{}번째'.format(i),'query clustering중..' ,'/', '총 {} 개'.format(len(query_label)))
        query_cluster = cluster_assignment[np.where(data_label==query)]
        indexs = np.where(cluster_assignment==query_cluster)
        result_list.append((query,data_label[indexs].tolist()))

    return result_list

# class FaissKMeans:
#     def __init__(self, n_clusters=8640, n_init=10, max_iter=300, query_result=None):
#         self.n_clusters = n_clusters
#         self.n_init = n_init
#         self.max_iter = max_iter
#         self.kmeans = None
#         self.cluster_centers_ = None
#         self.inertia_ = None
#         self.query_result = query_result
#     def fit(self, X):
#         self.kmeans = faiss.Kmeans(d=X.shape[1],
#                                 k=self.n_clusters,
#                                 niter=self.max_iter,
#                                 nredo=self.n_init,
#                                 gpu=True)
#         self.kmeans.train(X.astype(np.float32),init_centroids=self.query_result)
#         self.cluster_centers_ = self.kmeans.centroids
#         self.inertia_ = self.kmeans.obj[-1]

#     def predict(self, X):
#         return self.kmeans.index.search(X.astype(np.float32), 1)[1]