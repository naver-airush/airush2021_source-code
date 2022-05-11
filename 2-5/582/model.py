import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
os.environ["HF_HOME"] = "/home/nsml/.cache/huggingface"
import nsml
# from sklearn.model_selection import train_test_split
from larva import LarvaTokenizer, LarvaModel
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from arcface import ArcMarginProduct

class LarvaFeat(nn.Module):    
    def __init__(self):
        super().__init__()
        model_name = 'larva-kor-plus-base-cased'
        # model_name = 'gpt3-small-hyperclova'
        # model_name = 'r-larva-kor-large-cased'
        # model_name = 'alarva-electra-small-l2-hanakma-64k'
        # model_name = 'gpt3-medium-hyperclova'
        # model_name = 'gpt3-large-hyperclova'
        # model_name = 'distil_v2-r-larva-kor-small-cased'
        self.larva_model = LarvaModel.from_pretrained(model_name)
        self.tokenizer = LarvaTokenizer.from_pretrained(model_name)
        self.max_len = 32
        # self.fc = nn.Linear(self.larva_model.config.hidden_size, 768)
        # self.bn = nn.BatchNorm1d(768)
        self.fc = nn.Linear(self.larva_model.config.hidden_size, 512)
        self.bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(self.larva_model.config.hidden_size, 21332)
        self.bn2 = nn.BatchNorm1d(21332)
        self.model2 = None
        self._init_params()
        self.arc = ArcMarginProduct(self.larva_model.config.hidden_size, 21332,
                                          s=30, m=0.5, easy_margin=False, ls_eps=0)
        # self.fc2_2 = nn.Linear(self.larva_model.config.hidden_size + 4, 21332)
        # self.bn3 = nn.BatchNorm1d(1)
        # self.fc3 = nn.Linear(1, 4)
        # self.dropout = nn.Dropout(p=0.3)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)
        

    def forward(self, x, x2=None):
        tokenizer_output = self.tokenizer(x, truncation=True, padding=True, max_length=self.max_len)
        if 'token_type_ids' in tokenizer_output:
            input_ids = torch.LongTensor(tokenizer_output['input_ids']).to('cuda')
            token_type_ids = torch.LongTensor(tokenizer_output['token_type_ids']).to('cuda')
            attention_mask = torch.LongTensor(tokenizer_output['attention_mask']).to('cuda')
            x = self.larva_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        else:
            input_ids = torch.LongTensor(tokenizer_output['input_ids']).to('cuda')
            attention_mask = torch.LongTensor(tokenizer_output['attention_mask']).to('cuda')
            x = self.larva_model(input_ids=input_ids, attention_mask=attention_mask)
        x = torch.sum(x.last_hidden_state * attention_mask.unsqueeze(-1), dim=1) / attention_mask.sum(dim=1, keepdims=True)
        if x2 is not None:
            # tokenizer_output2 = self.tokenizer(x2, truncation=True, padding=True, max_length=self.max_len)
            # input_ids2 = torch.LongTensor(tokenizer_output2['input_ids']).to('cuda')
            # token_type_ids2 = torch.LongTensor(tokenizer_output2['token_type_ids']).to('cuda')
            # attention_mask2 = torch.LongTensor(tokenizer_output2['attention_mask']).to('cuda')
            # x2 = self.larva_model(input_ids=input_ids2, token_type_ids=token_type_ids2, attention_mask=attention_mask2)
            # x2 = torch.sum(x2.last_hidden_state * attention_mask2.unsqueeze(-1), dim=1) / attention_mask2.sum(dim=1, keepdims=True)
            # x2 = self.fc3(self.bn3(x2.unsqueeze(dim=1)))
            # x = torch.cat([x, x2], dim=1)
            # x = self.fc2_2(x)
        # x = self.fc2(self.dropout(x))
            x = self.arc(x, x2)
            return x

        else:
            x = self.fc2(x)
            logits = self.bn2(x)
            return logits
        # return x
    
    def extract_feat(self, x):
        tokenizer_output = self.tokenizer(x, truncation=True, padding=True, max_length=self.max_len)
        if 'token_type_ids' in tokenizer_output:
            input_ids = torch.LongTensor(tokenizer_output['input_ids']).to('cuda')
            token_type_ids = torch.LongTensor(tokenizer_output['token_type_ids']).to('cuda')
            attention_mask = torch.LongTensor(tokenizer_output['attention_mask']).to('cuda')
            x = self.larva_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        else:
            input_ids = torch.LongTensor(tokenizer_output['input_ids']).to('cuda')
            attention_mask = torch.LongTensor(tokenizer_output['attention_mask']).to('cuda')
            x = self.larva_model(input_ids=input_ids, attention_mask=attention_mask)

        x = torch.sum(x.last_hidden_state * attention_mask.unsqueeze(-1), dim=1) / attention_mask.sum(dim=1, keepdims=True)
        x = self.fc(x)
        x = self.bn(x)
        return x

    # def tfidf(self, dataset):
    #     max_features = 2**12
    #     self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,3), tokenizer=self.tokenizer.tokenize)
    #     self.pca = PCA(n_components=0.95, random_state=42)
    #     data = dataset[:]['prod_nm'].values
    #     v_data = self.vectorizer.fit_transform(data)
    #     print('tfidfvectorizer train complete')
    #     print('v_data', v_data[:10])
    #     train_reduced= self.pca.fit_transform(v_data.toarray())
    #     print('pca train complete')
    #     print('train_reduced', train_reduced[:10])
    #     self.model2 = [self.vectorizer, self.pca]

