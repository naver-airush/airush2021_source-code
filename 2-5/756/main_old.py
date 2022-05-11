import os
import argparse
import torch
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses
from torch.utils.data import DataLoader
import nltk
from model import MyModel
from dataset import CatalogDataset
from predict import predict , predict_infer, result_maker, k_mean_result_maker, l2_result_maker
from torch.utils.data import DataLoader
from train import train, test, train_iter, train_wd
import time
from gensim.models import Word2Vec
import re
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

        # implement inference code here with the trained model
        # returns list of (query_nv_mid,  [database_nv_mid])
        # result = [('83005822834', ['83005822834', '25859958427', '24573884420', '23750705737', '24268336001', '17865605897', '21031068803', '82161730914', '82670588319', '82382265727', '11634617066', '20978096318', '25412085466', '7135606589', '13318197891', '82402979262', '23692422430', '9592722787', '24304299825', '22723211067', '13307168394', '8790672801', '10022985319', '24835333791', '20904167870', '12938921457', '81260282761', '13333122455', '81175101'])]
        
        return result
        # return [('1906368762', ['1906368762','1810466025','5159532445']),
        #         ('636762', ['636762','1146025','155245'])] # dummy result

    nsml.bind(save, load, infer)

def isHangul(text):
    encText = text
    hanCount = len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', encText))
    return hanCount > 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=5)
    
    args = parser.parse_args()
    
    # model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    ## model define ##
    nltk.download('punkt')
    # model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    # model_name = 'sentence-transformers/paraphrase-xlm-r-multilingual-v1'
    model_name = 'bert-base-multilingual-uncased'
    # model_name = 'bert-base-multilingual-cased'

    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=False, pooling_mode_cls_token=True, pooling_mode_max_tokens=False)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    if IS_ON_NSML:
        bind_nsml(model, args)
        # DONOTCHANGE: They are reserved for nsml
        # Warning: Do not load data before the following code!
        
        if args.pause:
            nsml.paused(scope=locals())


    dataset = CatalogDataset(args,os.path.join(DATASET_PATH, 'train', 'train_data'))

    total_sentences = dataset.train_sentences()
    total_labels = dataset.train_labels()
    
    total_length = len(total_sentences)
    valid_length = 1000

    train_sentences = total_sentences[:total_length-valid_length]
    train_labels = total_labels[:total_length-valid_length]    

    valid_sentences = total_sentences[total_length-valid_length:]
    valid_labels = total_labels[total_length-valid_length:]



    # nsml.load(checkpoint='2',  session="KR95459/airush2021-2-5/297")
    if args.mode == 'train':
        
        nsml.save('first')
        # len(dataset) ## 90500 / 8640 으로 나눠야함
        # train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset)-8640, 8640])
        # train_set, real_train_set = torch.utils.data.random_split(dataset, [len(dataset)-90516, 90516])
        # train_set_loader= DataLoader(real_train_set, args.batch_size)
        # val_set_loader= DataLoader(val_set, args.batch_size)
        # result,label = predict(args,model,train_set_loader) #database
        # result2,label2 = predict(args,model,val_set_loader) # query
        # print(len(result))
        # print(len(label))

        ##training 시킬때..

        query_labels, query_sentences, count_dict= query_finder(valid_labels,valid_sentences)
        train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)
        valid_dataset =datasets.DenoisingAutoEncoderDataset(valid_sentences)
        model.to('cuda')

        warmup=False
        epoch = 5
        name_of_model = 'TSDAE'
        for epoch in range(args.epoch):
            model = train_iter(args,epoch,model,train_dataset,train_sentences,train_labels,name_of_model,warmup)
        test(model,query_labels,query_sentences,valid_labels,valid_sentences,count_dict)
        # name_of_model = 'TSDAE'
        # ##training 시킬때..
        # warmup=False
        # for epoch in range(args.epoch):
        #     model = train_iter(args,epoch,model,train_dataset,train_sentences,train_labels,name_of_model,warmup)
        #     test(model,query_labels,query_sentences,valid_labels,valid_sentences,count_dict)
        #     warmup = False

        # test(model,query_labels,query_sentences,valid_labels,valid_sentences,count_dict)

        # for i , valid in enumerate(valid_sentences):
        #     print(valid,'[{}]'.format(valid_labels[i]))
        # ## ['고릴라가 드럼을 친다' ~~~...] valid set 따로 만들어서 이걸로 순위 재서 test하기
        nsml.save('final')

def query_finder(labels,sentences):
    dict = {}
    count_dict = {}
    for index, label in enumerate(labels):
        try:
            dummy = dict[label]
            count_dict[label] += 1
        except:
            dict[label] = sentences[index]
            count_dict[label] = 1

    query_labels = []
    query_sentences = []

    for label, sentence in dict.items():
        query_labels.append(label)
        query_sentences.append(sentence)

    return query_labels, query_sentences, count_dict





        # result = result_maker(result,label,result2,label2)

        # print(result)



if __name__ == "__main__":
    main()