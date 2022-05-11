#nsml: pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
from distutils.core import setup

setup(
    name='Catalog matching model',
    version='0.2',
    description='Catalog matching model',
    install_requires=[
        'numpy',
        'python-snappy==0.6.0',
        'pyarrow==2.0.0',
        'fastparquet==0.4.2',
        'pandas==1.1.5',
        # 'sentence_transformers',
        # 'faiss-gpu',
        'gensim'

        # 'kmeans-pytorch',
        #'gluonnlp',
        #'mxnet'

        #'transformers==3.5',
        #'albumentations',
        #'faiss',
        #'sklearn',
        #'opencv-python',
    ],
)