import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import argparse
import torch
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses
from sentence_transformers.readers import STSBenchmarkDataReader, InputExample
from torch.utils.data import DataLoader
import nltk
from model import MyModel
from dataset import CatalogDataset, query_finder
from predict import predict , predict_infer, result_maker, k_mean_result_maker, l2_result_maker, result_maker_for_test
from torch.utils.data import DataLoader
from train import train, test, train_iter, train_iter_STS, train_iter_STS_warmup
import time
from gensim.models import Word2Vec
import re
import numpy as np
from evaluation import evaluate
# from kmeans_pytorch import kmeans

import nsml
from nsml import DATASET_PATH, IS_ON_NSML

import os
import subprocess

def bind_nsml(model, args):
    def save(path):
        print('save: path=', path)
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), open(os.path.join(path, 'model.pt'), 'wb'))
        print('model saved')

    def load(path):
        print('load: path=', path)
        model.load_state_dict(torch.load(open(os.path.join(path, 'model.pt'), 'rb')))
        print('model loaded')

    def infer(dataset_path):
        print('infer: dataset_path=', dataset_path)

        database_path = os.path.join(dataset_path, 'test', 'test_data', 'database')
        query_path = os.path.join(dataset_path, 'test', 'test_data', 'query')
        database_dataset = CatalogDataset(args,database_path, has_label=False)
        query_dataset = CatalogDataset(args, query_path, has_label=False)
        train_sentences = query_dataset.train_sentences()

        print('query_length:',len(query_dataset))
        print('queries')
        for query in train_sentences:
            print(query)

        databese_loader = DataLoader(database_dataset, args.batch_size)
        query_loader = DataLoader(query_dataset, args.batch_size)

        data_result,data_label = predict_infer(args,model,databese_loader)
        query_result,query_label = predict_infer(args,model,query_loader)

        # result = result_maker(data_result,data_label,query_result,query_label)
        result = result_maker(data_result,data_label,query_result,query_label)
        
        print(result)

        return result

    nsml.bind(save, load, infer)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=100)
    
    args = parser.parse_args()

    model_path = './output/training_sts'

    model = SentenceTransformer(model_path)

    if IS_ON_NSML:
        bind_nsml(model, args)
        # DONOTCHANGE: They are reserved for nsml
        # Warning: Do not load data before the following code!
        if args.pause:
            nsml.paused(scope=locals())


    nsml.load(checkpoint='94',  session="KR95459/airush2021-2-5/725")
    # nsml.load(checkpoint='21',  session="KR95459/airush2021-2-5/715")
    # nsml.load(checkpoint='48',  session="KR95459/airush2021-2-5/723")
    dataset = CatalogDataset(args,os.path.join(DATASET_PATH, 'train', 'train_data'))

    total_sentences = dataset.train_sentences()
    total_labels = dataset.train_labels()
    total_nv_mids = dataset.get_nv_mid()
    
    total_length = len(total_sentences)
    valid_length = 1000

    train_sentences = total_sentences[:total_length-valid_length]
    train_labels = total_labels[:total_length-valid_length]    
    train_nv_mids = total_nv_mids[:total_length-valid_length]   

    valid_sentences = total_sentences[total_length-valid_length:]
    valid_labels = total_labels[total_length-valid_length:]
    valid_nv_mids = total_nv_mids[total_length-valid_length:]

    if args.mode == 'train':
        nsml.save('king_of_ai')
        sys.exit(0)
        query_labels, query_sentences,query_nv_mids, count_dict= query_finder(train_labels,train_sentences,train_nv_mids)
        v_query_labels, v_query_sentences, v_query_nv_mids, v_count_dict= query_finder(valid_labels,valid_sentences,valid_nv_mids)
        gt_list = []
        for nv_mid in v_query_nv_mids: #여기서 label은 카탈로그 label
            gt_list.append((nv_mid,[]))
        for e1,valid_label in enumerate(valid_labels):
            valid_nv_mid = valid_nv_mids[e1]
            for e2,v_query_label in enumerate(v_query_labels): ##여기서 label은 카탈로그 label
                if valid_label == v_query_label: ## if valid_label == gt_label
                    gt_list[e2][1].append(valid_nv_mid)

        model.to('cuda')
        InputExample_list = InputExample_maker(model,query_labels,query_sentences,count_dict,train_labels,train_sentences)
        # sys.exit(1)
        batch_size = args.batch_size
        warmup = False
        test(args,model,v_query_labels, v_query_sentences,valid_labels,valid_sentences,v_count_dict,0)
        threshold_test(args,model,v_query_sentences,v_query_nv_mids,valid_sentences,valid_nv_mids,0,gt_list)
        first = True
        fix_token = True
        for epoch in range(args.epoch):
            if first:
                first = False
                pass
            else:
                if fix_token:
                    if epoch%5==0:
                        InputExample_list = InputExample_maker(model,query_labels,query_sentences,count_dict,train_labels,train_sentences)
                        fix_token = True #False면 Inputexample 하나로 fix하는 것
                print('STS {}epoch 학습 중'.format(epoch))
                print("총 dataset 개수: ", len(InputExample_list))
                print("batch_size: ", batch_size)
                print("총 iteration: ", len(InputExample_list)//batch_size)
                model.train()
                if warmup == True:
                    model = train_iter_STS_warmup(args,epoch,batch_size,model,InputExample_list)
                    warmup=False
                else:
                    model = train_iter_STS(args,epoch,batch_size,model,InputExample_list) ## 여기서부터 고치자
                threshold_test(args,model,v_query_sentences,v_query_nv_mids,valid_sentences,valid_nv_mids,epoch,gt_list)
                test(args,model,v_query_labels, v_query_sentences,valid_labels,valid_sentences,v_count_dict,epoch)

            time.sleep(30)



        nsml.save('final')

def InputExample_maker(model,query_labels,query_sentences,count_dict,train_labels,train_sentences):
    model.to('cuda')
    model.eval()
    InputExample_list = []
    with torch.no_grad():
        queries = query_sentences
        corpus = train_sentences
        corpus_labels = train_labels
        corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
        print("Query 개수: ", len(query_sentences))
        positive_example = 0
        negative_example = 0
        for i,query in enumerate(queries):
            top_k = count_dict[query_labels[i]]*2 ##예상 correct 개수의 2배
            query_embedding = model.encode(query, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
            cos_scores = cos_scores.cpu()
            top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
            for e , idx in enumerate(top_results[0:top_k]): # top k의 copus index들 
                if e == top_k//2:
                    ## query 개수만큼 X 2배
                    for j in range(e):
                        if query_labels[i] == corpus_labels[top_results[e-(j+1)]]: # label 같아야 Inputexample 취합
                            if cos_scores[ top_results[e-(j+1)] ] < 0.1:
                                pass
                            else:
                                InputExample_list.append(InputExample(texts=[ query, corpus[ top_results[e-(j+1)] ] ],label= min(cos_scores[ top_results[e-(j+1)] ] + 0.2, 0.999 )))
                                positive_example+=1
                            #위에 max이면 극단적 labelling
                        else:
                            #밑에 min이면 극단적 labelling
                            if cos_scores[ top_results[e-(j+1)] ] > 0.9:
                                pass
                            else:
                                InputExample_list.append(InputExample(texts=[ query, corpus[ top_results[e-(j+1)] ] ],label= max(cos_scores[ top_results[e-(j+1)] ]-0.2, 0.001 )))
                                negative_example+=1

                        if query_labels[i] != corpus_labels[top_results[e+j]]: # label 달라야 Inputexample 취합
                            if cos_scores[ top_results[e+j] ] > 0.9:
                                pass
                            else:
                                InputExample_list.append(InputExample(texts=[ query, corpus[ top_results[e+j] ]],label= max(cos_scores[ top_results[e+j] ]-0.2 ,  0.001) ) )
                                negative_example+=1
                            #위에 min이면 극단적 labelling
                        else:
                            #밑에 max이면 극단적 labelling
                            if cos_scores[ top_results[e+j] ] < 0.1:
                                pass
                            else:
                                InputExample_list.append(InputExample(texts=[ query, corpus[ top_results[e+j] ]],label= min(cos_scores[ top_results[e+j] ] + 0.2 ,  0.999) ) )
                                positive_example+=1
                        if j == 3: ## 1/1
                            break
                        # if j == 3: ## 3/3
                        #     break
                    break ## !!!! 필수 !!!! ##                
                
                ## 1/1 할 때 필수
                # if query_labels[i] == corpus_labels[idx]:
                #     pass

                # else:
                #     # 원래점수 고려 labeling
                #     if cos_scores[top_results[e-1]] > 0.1:
                #         InputExample_list.append(InputExample(texts=[query, corpus[ top_results[e-1] ]],label= min(cos_scores[top_results[e-1]]+0.1 ,  0.99  ) )  ) 
                #     if cos_scores[top_results[e]] < 0.9:
                #         InputExample_list.append(InputExample(texts=[query, corpus[ top_results[e] ]],label= max(cos_scores[top_results[e]]-0.1  ,  0.01  )  )  )
                #     break
                    # 원래점수 고려 labeling
                ## 1/1 할 때 필수


            print("{}번째/{} query 처리중".format(i,len(queries)))
    print("총 dataset 개수: ", len(InputExample_list))
    print("positve example 비율: ", round(positive_example/len(InputExample_list),4))
    print("negative example 비율: ", round(negative_example/len(InputExample_list),4))
    
    return InputExample_list



def threshold_test(args,model,v_query_sentences,v_query_nv_mids,valid_sentences,valid_nv_mids,epoch,gt_list):
    print('threshold test start!!')
    model.eval()
    best_score = -1
    with torch.no_grad():
        v_query_result = model.encode(v_query_sentences)
        valid_result = model.encode(valid_sentences)
    threshold_list = [0.5+0.01*x for x in range(40)]
    print("epoch:{} threshold_test 시작".format(epoch))
    for threshold in threshold_list:
        result = result_maker_for_test(valid_result,valid_nv_mids,v_query_result,v_query_nv_mids,threshold=threshold)
    ## 추측 list
        score = evaluate(result,gt_list)
        print('threshold:{}'.format(threshold),'Score:{}'.format(score))
        best_score = max(score,best_score)
        if score == best_score:
            dict_ = {best_score:threshold}
    print('threshold: ',dict_[best_score],'best_score:',best_score)
    nsml.report(summary=True, step=epoch, epoch_total=args.epoch,\
        best_threshold=float(dict_[best_score]),\
        best_score = float(best_score))




if __name__ == "__main__":
    main()