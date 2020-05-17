import click
import nlp
import pickle
import os
import zipfile
import logging
import json
import numpy as np

from tqdm.auto import tqdm
from math import log
from urllib.request import urlretrieve
from collections import Counter

from .dam.__main__ import dam as dam_cli
from .rnn.__main__ import rnn as rnn_cli

logging.basicConfig(level=logging.INFO)
logger = logging


@click.group()
def cli():
    pass


# python -m spacy download en

@cli.command()
def prepare_dataset():
    from .data_model import ProcessedExample

    logger.info('Loading dataset.')
    dataset = nlp.load_dataset('snli')
    df = Counter()
    ex_count = 0
    os.makedirs('data', exist_ok=True)
    for dataset_name in dataset:
        logger.info(f'Processing {dataset_name} dataset.')
        with open(f'data/{dataset_name}.jsonl', 'w') as fw:
            for ex in tqdm(dataset[dataset_name]):
                ex_count += 1
                ex = ProcessedExample(**ex)
                unique_tokens = set(ex.premise + ex.hypothesis)

                for t in unique_tokens:
                    if dataset_name != 'train':
                        if t not in df:
                            df[t] = 0
                    else:
                        df[t] += 1
                fw.write(json.dumps(dict(ex)) + '\n')

    df = {t: log(ex_count / (df[t] + 1)) for t in df}

    logger.info(f'Storing vocab with stats.')
    with open('data/vocab.pkl', 'wb') as fw:
        pickle.dump(df, fw)


@cli.command()
def download_glove_embeddings():
    os.makedirs('data', exist_ok=True)
    glove_link = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
    download_path = 'data/glove.840B.300d.zip'

    logger.info(f'Downloading embeddings.')
    urlretrieve(glove_link, download_path)

    logger.info(f'Unzipping embeddings.')
    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall('data')


@cli.command()
def prepare_embeddings():
    logger.info(f'Loading vocab.')
    with open('data/vocab.pkl', 'rb') as fr:
        vocab = list(pickle.load(fr).keys())

    logger.info(f'Creating decomposed embeddings.')
    word_vectors = []
    i2w = []
    with open('data/glove.840B.300d.txt', encoding="utf-8") as fr:
        for line in tqdm(fr):
            split_line = line.split()
            if len(split_line) > 301:
                continue
            if split_line[0] in vocab:
                word_vectors.append([float(v) for v in split_line[1:]])
                i2w.append(split_line[0])
    word_vectors = np.array(word_vectors)
    word_vectors = np.vstack((
        np.array([[0.] * len(word_vectors[0])]),
        word_vectors.mean(axis=0, keepdims=True),
        word_vectors
    ))
    i2w = ['<PAD>'] + ['<UNK>'] + i2w
    w2i = {w: i for i, w in enumerate(i2w)}

    logger.info(f'Storing decomposed embeddings.')
    with open('data/decomposed_embeddings.pkl', 'wb') as fw:
        pickle.dump((np.array(word_vectors), w2i, i2w), fw)


# cli.add_command(centroid)
cli.add_command(rnn_cli)
cli.add_command(dam_cli)

if __name__ == '__main__':
    cli()
