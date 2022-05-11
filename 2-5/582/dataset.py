import os
import pandas as pd
#import fastparquet
os.environ["HF_HOME"] = "/home/nsml/.cache/huggingface"
import numpy as np
import random
import time
from collections import defaultdict
# from larva import LarvaTokenizer, LarvaModel
from glob import glob
import torch
from torch.utils.data import DataLoader
from typing import Dict
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# tokenizer = LarvaTokenizer.from_pretrained('larva-kor-plus-base-cased', do_lower_case=False)
def get_data_columns(has_label):
    if has_label:
        return ['nv_mid', 'prod_nm', 'match_nv_mid']
    else:
        return ['nv_mid', 'prod_nm']


def read_product_data_from_parquet(parquet_path, has_label):
    df = pd.read_parquet(parquet_path, columns=get_data_columns(has_label))
    return df

def clean_text(sentence):
    sentence2 = re.sub('[^\w\d\-]', ' ', sentence.lower())
    return re.sub(' +', ' ', sentence2)
def get_catalogs(df):
    return list(set(df['match_nv_mid']))

def measure_split(sentence): 
    measurement_cats = ['kg', 'g', 'mg', 'ml', 'l']
    for m in measurement_cats: 
        pat = '(\d+)' + m + ''
        res = re.findall(pat, sentence)
        if res != []: 
            return f' {m} '.join(res) + ' ' + m

    for m in measurement_cats: 
        pat = '(\d+) ' + m + ''
        res = re.findall(pat, sentence)
        if res != []: 
            return f' {m} '.join(res) + ' ' + m
    return False
        
        
def measure_delete(sentence):
    measurement_cats = ['kg', 'g', 'mg', 'ml', 'l']
    for m in measurement_cats: 
        pat = '(\d+)' + m + ''
        sentence = re.sub(pat, '',sentence)

    for m in measurement_cats: 
        pat = '(\d+) ' + m + ''
        sentence = re.sub(pat, '',sentence)
    return sentence

def clean_text2(sentence):
    # sentence = re.sub('_', '', sentence)
    pat = '([a-z]+)' + '(\d+)'
    sentence = re.sub(pat, '', sentence)
    # sentence = re.sub('([a-z]+)', '', sentence)
    return sentence


# def to_token(df):
#     token_df = tokenizer.tokenize(df)
#     # token_ids = tokenizer.convert_tokens_to_ids(token_df)
#     return token_df

# def pad_to_ids(token):
#     ids = tokenizer.convert_tokens_to_ids(token)
#     max_len = 30
#     if len(ids) < max_len:
#         ids = ids + [0] * (max_len - len(ids))
#     else:
#         ids = ids[:max_len]
#     return ids

class CatalogDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, has_label=True):
        super(CatalogDataset, self).__init__()
        print('CatalogDataset: data_path=', data_path)

        self.data_path = data_path
        self.df = read_product_data_from_parquet(data_path, has_label)
        self.df['prod_nm'] = self.df['prod_nm'].apply(clean_text)
        self.df['measurement'] = self.df['prod_nm'].apply(measure_split)
        # self.df['prod_nm'] = self.df['prod_nm'].apply(clean_text2)
        # self.df['prod_nm'] = self.df['prod_nm'].apply(measure_delete)
        # self.df['prod_nm'] = self.df['prod_nm'].apply(to_token)
        # self.df['ids'] = self.df['prod_nm'].apply(pad_to_ids)

    def __len__(self):
        return self.df.shape(0)

    def __getitem__(self, idx):
        return self.df.iloc[idx]

    def cat_id2ids(self, cat_id):
        return self.label_id_dict.get(cat_id)


    def train_valid(self):
        le = LabelEncoder()
        # le2 = LabelEncoder()
        # label_info = list(set(self.df['match_nv_mid'].to_list()))
        self.df['label'] = le.fit_transform(self.df['match_nv_mid'])
        # self.df['measurement'] = le2.fit_transform(self.df['measurement'])
        # self.label_id_dict = {str(cat_id): i for i, cat_id in enumerate(label_info)}
        # self.df['label'] = self.df['match_nv_mid'].apply(self.cat_id2ids)
        print('num_class:', len(self.df['label'].value_counts()))
        print(self.df['label'].describe())
        train, valid = train_test_split(self.df, test_size=0.1, random_state=42)
        train = train.reset_index()
        valid = valid.reset_index()
        train.drop(['index'], axis = 1, inplace = True)
        valid.drop(['index'], axis = 1, inplace = True)
        return [train, valid]


