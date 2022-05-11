import os
import argparse
import torch
import sys



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=5)
    
    args = parser.parse_args()

    os.system('git clone https://github.com/SKTBrain/KoBERT.git')
    os.chdir('./KoBERT')
    os.system('pip install -r requirements.txt')
    os.system('pip install .')
    os.chdir('..')
    os.system('git clone https://github.com/billyirrish/KoSentenceBERT_SKTBERT.git')
    os.chdir('./KoSentenceBERT_SKTBERT')
    os.system('pip install -r requirements.txt')
    os.system('wget https://github.com/kdh4672/Study_GAN/releases/download/1/result.pt')
    os.system('mkdir ./output/training_sts/0_Transformer')
    os.system('mv result.pt ./output/training_sts/0_Transformer/result.pt')
    # print(os.listdir('./KoSentenceBERT_SKTBERT/output/training_sts/0_Transformer/'))
    os.system('mv ../main2.py ./main2.py')
    os.system('python ./main2.py --pause {} --mode {}'.format(args.pause, args.mode))

if __name__ == "__main__":
    main()

