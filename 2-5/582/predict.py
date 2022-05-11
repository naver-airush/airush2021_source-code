import faiss
import torch.nn as nn
import torch
import numpy as np
from torch.nn.functional import threshold
import gc
import torch.nn.functional as F

device = "cuda:0"
class Comparing:
    def __init__(self, database, query, model):
        # self.vectorizer = model.model2[0]
        # self.pca = model.model2[1]
        self.model = model.to(device)
        self.db = database
        self.query = query
        self.data_loaders = {
            'db': torch.utils.data.DataLoader(self.db[:]['prod_nm'], batch_size=100, shuffle=False,drop_last=False, pin_memory=False, num_workers=0),
            'query': torch.utils.data.DataLoader(self.query[:]['prod_nm'], batch_size=100, shuffle=False,drop_last=False, pin_memory=False, num_workers=0)
        }
    
    # def query_expansion(self, feats, sims, topk_idx, thres=0.55):
    #     sims[:,0]=1
    #     weights = np.expand_dims(sims[:, :30], axis=-1).astype(np.float32)
    #     weights = np.where(weights>=thres, weights, 0)
    #     feats = (feats[topk_idx[:, :30]] * weights).sum(axis=1)
    #     # feats = (feats[topk_idx[:, :100]] * sims).sum(axis=1)
    #     return feats

    # def blend_neighborhood(self, emb, match_index_lst, similarities_lst):
    #     new_emb = emb.copy()
    #     for i in range(emb.shape[0]):
    #         cur_emb = emb[match_index_lst[i]]
    #         weights = np.expand_dims(similarities_lst[i], 1)
    #         new_emb[i] = (cur_emb * weights).sum(axis=0)
    #     new_emb = F.normalize(new_emb, axis=1)
    #     return new_emb

    def predict(self):
        db_feats = []
        query_feats = []
        for i, db_sentence in enumerate(self.data_loaders['db']):
            with torch.no_grad():
                db_feats_minibatch = self.model.extract_feat(db_sentence)
                db_feats.append(db_feats_minibatch.cpu().numpy())
        db_feats = np.concatenate(db_feats)
        db_feats /= np.linalg.norm(db_feats, 2, axis=1, keepdims=True)
        for i, query_sentence in enumerate(self.data_loaders['query']):
            with torch.no_grad():
                query_feats_minibatch = self.model.extract_feat(query_sentence)
                query_feats.append(query_feats_minibatch.cpu().numpy())
        query_feats = np.concatenate(query_feats)
        # query_feats /= np.linalg.norm(query_feats, 2, axis=1, keepdims=True)
        print(db_feats[:10])
        res = faiss.StandardGpuResources()
        print('db_feats shape:', db_feats.shape)
        index = faiss.index_factory(db_feats.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
        # index = faiss.IndexFlatL2(db_feats[1])
        index = faiss.index_cpu_to_gpu(res, 0, index)
        db_feats = np.ascontiguousarray(db_feats, dtype=np.float32)
        query_feats = np.ascontiguousarray(query_feats, dtype=np.float32)
        faiss.normalize_L2(x=db_feats)
        faiss.normalize_L2(x=query_feats)


        # # clus = faiss.Clustering(db_feats.shape[1], 8640)
        # # clus.seed = np.random.randint(42)
        # # clus.niter = 20
        # # clus.train(db_feats, index)
        # kmeans = faiss.Kmeans(d=db_feats.shape[1], k=8640, niter=300, nredo=10, gpu=True)
        # kmeans.train(db_feats)
        # # D, I = index.search(query_feats, 30)
        # D, I = kmeans.index.search(query_feats, 50)

        index.add(db_feats)
        
        # D, I = index.search(db_feats, 30)
        # # # D, I = index.range_search(db_feats, 0.55)
        # # new_feats = self.query_expansion(db_feats, D, I)
        # new_feats = self.blend_neighborhood(db_feats, I, D)
        # print('new feats complete')
        # del index
        # gc.collect()
        # index = faiss.index_factory(new_feats.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
        # index = faiss.index_cpu_to_gpu(res, 0, index)
        # faiss.normalize_L2(new_feats)
        # index.add(new_feats)

        D, I = index.search(query_feats, 100)

        result_dict = {}
        result = []
        print(D[:10])
        print(I[:10])
        thres = 0.72
        print(f'thres = {thres}')
        for i in range(query_feats.shape[0]):
            sub_result = []
            sub_check = []
            for dist, idx in zip(D[i], I[i]):
                if dist > thres:
                    if (self.db[idx]['measurement'] == self.query[i]['measurement']) or ((not self.db[idx]['measurement']) or (not self.query[i]['measurement'])):
                        sub_result.append(str(self.db[idx]['nv_mid']))
                    sub_check.append((dist, self.db[idx]['prod_nm']))
            result.append((str(self.query[i]['nv_mid']), sub_result))
            result_dict[str(self.query[i]['nv_mid'])] = result_dict.get(str(self.query[i]['nv_mid']), []) + sub_result
            if i<30:
                print(self.query[i]['prod_nm'], '=>',sub_check)

        # # # tfidf 추가
        # v_db = self.vectorizer.transform(self.db[:]['prod_nm'].values)
        # v_query = self.vectorizer.transform(self.query[:]['prod_nm'].values)
        # db_reduced = self.pca.transform(v_db.toarray())
        # query_reduced = self.pca.transform(v_query.toarray())
        # res2 = faiss.StandardGpuResources()
        # index2 = faiss.index_factory (db_reduced.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
        # index2 = faiss.index_cpu_to_gpu(res2, 0, index2)
        # db_reduced = np.ascontiguousarray(db_reduced, dtype=np.float32)
        # query_reduced = np.ascontiguousarray(query_reduced, dtype=np.float32)
        # faiss.normalize_L2(x=db_reduced)
        # faiss.normalize_L2(x=query_reduced)
        # index2.add(db_reduced)
        # D, I = index2.search(query_reduced, 100)
        # # result = []
        # for i in range(query_reduced.shape[0]):
        #     sub_result = []
        #     sub_check = []
        #     for dist, idx in zip(D[i], I[i]):
        #         if dist > 0.8:
        #             sub_result.append(str(self.db[idx]['nv_mid']))
        #             # sub_check.append((dist, self.db[idx]['prod_nm']))
        #     # result.append((str(self.query[i]['nv_mid']), sub_result))
        #     result.append((str(self.query[i]['nv_mid']), list(set(result_dict.get(str(self.query[i]['nv_mid']), []))|set(sub_result))))
        #     if i<30:
        #         print(self.query[i]['prod_nm'], '=>',sub_check)
        return result

