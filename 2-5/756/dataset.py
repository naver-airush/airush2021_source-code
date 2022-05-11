import os
import pandas as pd
#import fastparquet
import numpy as np
import random
import time
from collections import defaultdict

from glob import glob
import torch
from torch.utils.data import DataLoader
from typing import Dict
from string import punctuation


def get_data_columns(has_label):
    if has_label:
        return ['nv_mid', 'prod_nm', 'match_nv_mid']
    else:
        return ['nv_mid', 'prod_nm']


def read_product_data_from_parquet(parquet_path, has_label):
    df = pd.read_parquet(parquet_path, columns=get_data_columns(has_label))
    return df


def get_catalogs(df):
    return list(set(df['match_nv_mid']))

def query_finder(labels,sentences,nv_mid):
    dict = {}
    count_dict = {}
    for index, label in enumerate(labels):
        try:
            dummy = dict[label]
            count_dict[label] += 1
        except:
            dict[label] = [sentences[index],nv_mid[index]]
            count_dict[label] = 1

    query_labels = []
    query_sentences = []
    query_nv_mid = []

    for label, item_list in dict.items():
        query_labels.append(label)
        query_sentences.append(item_list[0])
        query_nv_mid.append(item_list[1])


    return query_labels, query_sentences, query_nv_mid, count_dict



class CatalogDataset(torch.utils.data.Dataset):
    def __init__(self, args, data_path, has_label=True):
        super(CatalogDataset, self).__init__()
        print('CatalogDataset: data_path=', data_path)
        self.batch_size = args.batch_size
        self.data_path = data_path
        self.df = read_product_data_from_parquet(data_path, has_label)
        self.delete_list = ['무료배송','배송','무료','핫딜','당일배송','빠른배송','퀵배송','당일','익일발송','발송']


    def train_sentences(self):
        # print("preprocessing 전:",self.df['prod_nm'].tolist()[0:10])
        x = self.pre_processing_list(self.df['prod_nm'].tolist())
        # print("preprocessing 후:",x[0:10])
        return x

    def train_labels(self):
        return self.df['match_nv_mid'].tolist()

    def get_nv_mid(self):
        return self.df['nv_mid'].tolist()

    def __len__(self):
        print(self.df.shape)
        return self.df.shape[0]

    def __getitem__(self, idx): 
        items = self.df.iloc[idx]
        if self.df.shape[1] == 3:
            # nv_mid = [items['nv_mid']]
            # prod_nm = [items['prod_nm']]
            # match_nv_mid = [items['match_nv_mid']]
            nv_mid = items['nv_mid']
            prod_nm = items['prod_nm']
            match_nv_mid = items['match_nv_mid']
            prod_nm = self.pre_processing(prod_nm)
            return nv_mid ,prod_nm, match_nv_mid

        else :
            nv_mid = items['nv_mid']
            prod_nm = items['prod_nm']
            prod_nm = self.pre_processing(prod_nm)
            return nv_mid ,prod_nm

    def get_torch_loader(self, dataset):

        loader = DataLoader(dataset, self.batch_size)

        return loader

    def pre_processing(self,prod_nm):
    
        for ele in prod_nm: 
            if ele in punctuation: 
                prod_nm = prod_nm.replace(ele, " ")
                
        seperator = " "
        words = prod_nm.split(seperator)

        for ele in words:
            if ele in self.delete_list:
                words.remove(ele)
        prod_nm = ' '.join(words) ## 띄어쓰기 없애고도 해보기
        return prod_nm

    def pre_processing_list(self,prod_nm_list):
        
        for i,prod_nm in enumerate(prod_nm_list):
            for ele in prod_nm: 
                if ele in punctuation: 
                    prod_nm = prod_nm.replace(ele, " ")
                    
            seperator = " "
            words = prod_nm.split(seperator)

            for ele in words:
                if ele in self.delete_list:
                    words.remove(ele)
            prod_nm = ' '.join(words) ## 띄어쓰기 없애고도 해보기

            prod_nm_list[i] = prod_nm

        return prod_nm_list

