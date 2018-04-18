from pytensor.utils.vocabulary import *
import os

def load_ptb():
    file_path = '../../data/corpus/ptb.train.txt'
    raw_sentences = open(file_path, 'r').readlines()

    sentences = []
    for raw_sentence in raw_sentences:
        words = raw_sentence.strip().split()
        sentences.append(words)

    vocab = create_vocabulary(sentences)

    id_sentences = []

    for sentence in sentences:
        id_sentence = vocab.get_ids(sentence)
        id_sentences.append(id_sentence)

    return id_sentences, vocab