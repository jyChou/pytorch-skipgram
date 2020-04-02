import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from os import path
import collections

# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

def read_file(data_path):
    ret = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            ret.append(line)
    return ret



def preprocess(data):
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    outputs = []
    idx = 0
    for sentence in data:
        lemmatizer = WordNetLemmatizer()
        # delete puncation
        sentence = re.sub(r'[^a-z ]', "", sentence.lower())
        # delete double space
        sentence = re.sub(r'[ ]+', " ", sentence)
        
        output = nltk.word_tokenize(sentence)
        #tags = nltk.pos_tag(output)
        #output = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in output]
        output = [lemmatizer.lemmatize(w) for w in output]
        outputs.append(output)
        idx += 1
        if idx % 500 == 0:
            print('Processing {}% of data'.format(idx/len(data)*100))
    return outputs

def count_vocab(data):
    counter = collections.defaultdict(int)
    for sentence in data:
        for w in sentence:
            counter[w] += 1
    return counter

def build_vocab(data, counter, threshold = 10):
    vocabs = set()
    for sentence in data:
        for w in sentence:
            if counter[w] > threshold:
                vocabs.add(w)
            else:
                vocabs.add('UNK')
    return {vocab:idx for idx, vocab in enumerate(vocabs)}, {idx: vocab for idx, vocab in enumerate(vocabs)}



def vectorize(data, vocab, window_size):
    ret = []
    for sentence in data:
        for i in range(len(sentence)):
            for j in range(-window_size, window_size+1):
                if j == 0 or i+j >= len(sentence) or i+j < 0:
                    continue
                else:
                    w1, w2 = sentence[i], sentence[i+j]
                    if w1 not in vocab:
                        w1 = 'UNK'
                    if w2 not in vocab:
                        w2 = 'UNK'
                    ret.append((vocab[w1], vocab[w2]))
    return ret
    
def load_data(data_path = '../data/training/training-data.1m', ws=2):
    
    if path.exists('idx_to_vocab_400_ds_5.npy') and path.exists('train_data_400_ds_5.npy'):
        train_data = np.load('train_data_400_ds_5.npy')
        idx_to_vocab = np.load('idx_to_vocab_400_ds_5.npy', allow_pickle = True)[()]
    else:
        train_data = read_file(data_path)
        train_data = preprocess(train_data)
        counter = count_vocab(train_data)
        vocab_to_idx, idx_to_vocab = build_vocab(train_data, counter)
        train_data = vectorize(train_data, vocab_to_idx, ws)
        np.save('idx_to_vocab', idx_to_vocab)
        np.save('train_data', train_data)
    return train_data, idx_to_vocab