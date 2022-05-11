# from sentence_transformers import SentenceTransformer, LoggingHandler, InputExample
# from sentence_transformers import models, util, datasets, evaluation, losses
from sentence_transformers.readers import STSBenchmarkDataReader, InputExample
from sentence_transformers import datasets,SentenceTransformer,  SentencesDataset, LoggingHandler, losses, models, util
from torch.utils.data import DataLoader
import nsml
import torch
import numpy as np
import torch.nn as nn
from itertools import combinations
import math
import time
# Define your sentence transformer model using CLS pooling


def train(args,model,train_sentences):
    # Define a list with sentences (1k - 100k sentences)
    # Create the special denoising dataset that adds noise on-the-fly
    print('epoch:',args.epoch)
    print('batch_size:',args.batch_size)
    train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)
    
    # DataLoader to batch your data
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Use the denoising auto-encoder loss
    train_loss = losses.DenoisingAutoEncoderLoss(model, tie_encoder_decoder=True)

    # Call the fit method
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epoch,
        weight_decay=0,
        scheduler='constantlr',
        optimizer_params={'lr': 3e-5},
        show_progress_bar=True
    )

def train_iter(args,epoch,model,train_dataset,train_sentences,train_labels,model_name,warmup,sts_first=False):
    # Define a list with sentences (1k - 100k sentences)
    # Create the special denoising dataset that adds noise on-the-fly

    if model_name == 'TSDAE':
        print('총 epoch:',args.epoch)
        print('현재 epoch:',epoch)
        print('batch_size:',args.batch_size)
        # DataLoader to batch your data
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        # Use the denoising auto-encoder loss
        train_loss = losses.DenoisingAutoEncoderLoss(model, tie_encoder_decoder=True)

        # Call the fit method
        num_warmup_steps = (len(train_sentences) // args.batch_size) * 2
        if warmup == True:
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=1,
                weight_decay=0,
                # scheduler = 'warmupcosine',
                scheduler = 'warmuplinear',
                # scheduler = 'constantlr',
                warmup_steps = num_warmup_steps,
                optimizer_params={'lr': 3e-5},
                show_progress_bar=True
            )
        else:
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=1,
                weight_decay=0,
                # scheduler = 'warmupcosine',
                # scheduler = 'warmuplinear',
                scheduler = 'constantlr',
                # warmup_steps = num_warmup_steps,
                optimizer_params={'lr': 3e-4},
                show_progress_bar=True
            )
            


    
    elif model_name == "SIMCSE":
        # filtered_sentences = []
        # dict = {}
        # for i,label in enumerate(train_labels):
        #     try: dict[label] +=1
        #     except:
        #         dict[label] = 1
        #         filtered_sentences.append(train_sentences[i])
        # print('train_sentences class 수:', len(dict))
        # print('filtered_sentences 수:', len(filtered_sentences))
        train_data = SentencesDataset([InputExample(texts=[s, s]) for s in train_sentences],model)
        train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
        train_loss = losses.MultipleNegativesRankingLoss(model)

        # Call the fit method
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            evaluator =None
            # show_progress_bar=True
        )

    # elif model_name == 'STS':
    #     if sts_first == True:
    #         InputExample_list = []
    #         query_labels, query_sentences, count_dict= query_finder(train_labels,train_sentences)
    #         with torch.no_grad():
    #             queries = query_sentences
    #             corpus = train_sentences
    #             top_k = 10
    #             corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    #             print("Query 개수: ", print(len(query_sentences)))
    #             for i,query in enumerate(queries):
    #                 top_k = count_dict[query_labels[i]] + 1
    #                 query_embedding = model.encode(query, convert_to_tensor=True)
    #                 cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    #                 cos_scores = cos_scores.cpu()
    #                 #We use np.argpartition, to only partially sort the top_k results
    #                 top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
    #                 for idx in top_results[0:top_k]:
    #                     if query_labels[i] == train_labels[idx]:
    #                         pass
    #                     else:
    #                         InputExample_list.append(InputExample(texts=[query, train_sentences[idx-1]],label=1))
    #                         InputExample_list.append(InputExample(texts=[query, train_sentences[idx]],label=0))
    #                         break
    #                 print("{}번째/{} query 처리중".format(i,len(queries)))
    #         print("총 dataset 개수: ", len(InputExample_list))
    #     ##training##
    #     model.train()
    #     train_data = SentencesDataset(InputExample_list,model)
    #     train_dataloader = DataLoader(train_data, shuffle=True, batch_size=128)
    #     train_loss = losses.CosineSimilarityLoss(model=model)
    #     model.fit(
    #         train_objectives=[(train_dataloader, train_loss)],
    #         scheduler = 'constantlr',
    #         # warmup_steps = num_warmup_steps,
    #         optimizer_params={'lr': 3e-5},
    #         epochs=1,
    #         evaluator=None
    #     )     

    nsml.save(epoch)
    
    return model
def train_iter_STS_warmup(args,epoch,batch_size,model,InputExample_list):
    train_data = SentencesDataset(InputExample_list,model)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    warmup_steps = len(InputExample_list)//batch_size
    print('warmup_steps =', warmup_steps)
    model.fit(train_objectives=[(train_dataloader, train_loss)],
        evaluator=None,
        epochs= 1,
        optimizer_params = {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False},
        scheduler = 'warmupconstant',
        warmup_steps=warmup_steps,
        save_best_model = False,
        )

    nsml.save(epoch)
    
    return model

def train_iter_STS(args,epoch,batch_size,model,InputExample_list):
    train_data = SentencesDataset(InputExample_list,model)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)
    # model.fit(
    #     train_objectives=[(train_dataloader, train_loss)],
    #     epochs=1,
    #     evaluator=None
    # )
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator =None,
        epochs = 1,
        optimizer_params = {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False},
    #     use_amp = True,
        save_best_model = False,
        scheduler = 'constantlr',
    #     show_progress_bar=True
    )
    nsml.save(epoch)
    
    return model


def test(args,model,query_labels,query_sentences,valid_labels,valid_sentences,count_dict,epoch):
    model.eval()
    with torch.no_grad():
        queries = query_sentences
        corpus = valid_sentences
        top_k = 10
        corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
        print("Query 개수: ", print(len(query_sentences)))

        incorrect_max_score_list = []
        correct_min_score_list = []

        hard_incorrect = 0
        for i,query in enumerate(queries):
            query_embedding = model.encode(query, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
            cos_scores = cos_scores.cpu()
            #We use np.argpartition, to only partially sort the top_k results
            top_k = count_dict[query_labels[i]] + 5
            top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
            print("\n======================")
            print("Query:", query, '[{}]'.format(query_labels[i]))
            print('이 쿼리의 동일 label 개수:{}'.format(count_dict[query_labels[i]]))
            # if i<10: 
            print("\nTop 10 most similar sentences in corpus:")
            incorrect_max_score = -1
            correct_min_score = -1
            ims = -1
            cms = -1
            first_incorrect = True
            for e,idx in enumerate(top_results[0:top_k]):
                if query_labels[i] == valid_labels[idx]:
                    answer = 'Correct!!'
                    correct_min_score = "(Score: %.4f)" % (cos_scores[idx])
                    cms = cos_scores[idx]
                else:
                    answer = 'Incorrect'        
                    if first_incorrect == True:
                        incorrect_max_score = "(Score: %.4f)" % (cos_scores[idx])
                        ims = cos_scores[idx]
                        first_incorrect = False
                        if e < count_dict[query_labels[i]]:
                            hard_incorrect+=1
                # if i<10:
                print(corpus[idx].strip(), '[{}]'.format(valid_labels[idx]), "(Score: %.4f)" % (cos_scores[idx]),answer)
            print("correct_min_score:",correct_min_score)    
            print("incorrect_max_score:",incorrect_max_score)
            print('\n')
            if incorrect_max_score == -1:
                pass
            else:
                correct_min_score_list.append(cms)
                if ims > 0.9: ## mislabelling 제거
                    pass
                else:
                    incorrect_max_score_list.append(ims)
        print("correct_min_score 평균:",round(np.mean(correct_min_score_list),4))   
        print("incorrect_max_score 평균:",round(np.mean(incorrect_max_score_list),4))
        print("correct중 min: ", np.min(correct_min_score_list))
        print("incorrect중 max: ", np.max(incorrect_max_score_list))
        print("incorrect가 correct보다 넘는 비율:",round(hard_incorrect/len(queries),4))
        print("\n======================")
        nsml.report(summary=True, step=epoch, epoch_total=args.epoch,\
        smallest_correct=float(round(np.min(correct_min_score_list),4)),\
        correct_min_mean=float(round(np.mean(correct_min_score_list),4)),\
        incorrect_max_mean=float(round(np.mean(incorrect_max_score_list),4)),\
        min_max_difference= float(round(np.mean(correct_min_score_list)-np.mean(incorrect_max_score_list),4)),\
        largest_incorrect=float(round(np.max(incorrect_max_score_list),4)),\
        hard_incorrect_ratio=float(round(hard_incorrect/len(queries),4)))
