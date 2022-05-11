import os
import argparse
import torch
os.environ["HF_HOME"] = "/home/nsml/.cache/huggingface"
# os.system('pip install faiss-cpu --no-cache')
from model import LarvaFeat
from dataset import CatalogDataset
# from larva import LarvaTokenizer, LarvaModel
import transformers
import nsml
from nsml import DATASET_PATH, IS_ON_NSML
from predict import Comparing
import pickle
import numpy as np
from trainer import Trainer
import random
import argparse

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# def bind_nsml(solver, args):
def bind_nsml(model, args):
    def save(path):
        print('save: path=', path)
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), open(os.path.join(path, 'model.pt'), 'wb'))
        pickle.dump(model.model2, open(os.path.join(path, 'model2.pt'), 'wb'))
        print('model saved')

    def load(path):
        print('load: path=', path)
        model.load_state_dict(torch.load(open(os.path.join(path, 'model.pt'), 'rb')))
        # torch.load(open(os.path.join(path, 'model.pt'), 'rb'))
        model.model2 = pickle.load(open(os.path.join(path, 'model2.pt'), 'rb'))
        print('model loaded')

    def infer(dataset_path):
        print('infer: dataset_path=', dataset_path)
        database_path = os.path.join(dataset_path, 'test', 'test_data', 'database')
        query_path = os.path.join(dataset_path, 'test', 'test_data', 'query')
        database_dataset = CatalogDataset(database_path, has_label=False)
        query_dataset = CatalogDataset(query_path, has_label=False)
        # comparing = Comparing(database_dataset, query_dataset, solver.model)
        comparing = Comparing(database_dataset, query_dataset, model)
        result = comparing.predict()
        # implement inference code here with the trained model

        # returns list of (query_nv_mid,  [database_nv_mid])
        # return [('1906368762', ['1906368762','1810466025','5159532445']),
        #         ('636762', ['636762','1146025','155245'])] # dummy result
        return result

    nsml.bind(save, load, infer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument('--pause', type=int, default=0)
    args = parser.parse_args()

    model = LarvaFeat()
    # solver = Cluster()

    if IS_ON_NSML:
        # bind_nsml(solver, args)
        bind_nsml(model, args)

        # DONOTCHANGE: They are reserved for nsml
        # Warning: Do not load data before the following code!
        if args.pause:
            nsml.paused(scope=locals())

    dataset = CatalogDataset(os.path.join(DATASET_PATH, 'train', 'train_data'))
    train, valid = dataset.train_valid()
    print('trainset:', train['prod_nm'][:5])
    print('validset:', valid['prod_nm'][:5])
    # print(train['measurement'][:5])
    # print(valid['measurement'][:5])
    if args.mode == 'train':
        # implement training code here
        # solver.train(dataset)
        # model.tfidf(dataset)
        model = model.to('cuda')
        Trainer(model, train, valid)
        nsml.save('final')

if __name__ == "__main__":
    main()