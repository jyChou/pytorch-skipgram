import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from tqdm import tqdm
from os import path
import multiprocessing
import time
import pickle
import collections
from nltk.corpus import stopwords

def load_data(data_path = '../data/training/training-data.1m', ws=1):
    
    if path.exists('idx_to_vocab.npy') and path.exists('train_data.npy'):
        train_data = np.load('train_data.npy')
        idx_to_vocab = np.load('idx_to_vocab.npy', allow_pickle = True)
    else:
        train_data = read_file(data_path)
        train_data = preprocess(train_data)
        vocab_to_idx, idx_to_vocab = build_vocab(train_data)
        train_data = vectorize(train_data, vocab_to_idx, ws)
        np.save('idx_to_vocab', idx_to_vocab)
        np.save('train_data', train_data)
    return train_data, idx_to_vocab

def load_csv(data_path):
    word_set = set()
    with open(data_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx != 0 :
                _, word1, word2 = line.strip('\n').split(',')
                word_set.add(word1)
                word_set.add(word2)
    return word_set

def read_file(data_path):
    ret = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            ret.append(line)
    return ret



def preprocess(data):
    stop_words = set(stopwords.words('english'))
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    outputs = []
    for idx, sentence in enumerate(data):
        lemmatizer = WordNetLemmatizer()
        # delete puncation
        sentence = re.sub(r'[^a-z ]', "", sentence.lower())
        # delete double space
        sentence = re.sub(r'[ ]+', " ", sentence)
        output = nltk.word_tokenize(sentence)
        output = [w for w in output if w not in stop_words]
        # tags = nltk.pos_tag(output)
        # output = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in output]
        # output = [lemmatizer.lemmatize(w) for w in output]
        outputs.append(output)
        # if idx % 5000 == 0:
        #   print('Processing {}% of data'.format(idx/len(data)*100))
    return outputs

def count_vocab(data):
    counter = collections.defaultdict(int)
    for sentence in data:
        for w in sentence:
            counter[w] += 1
    return counter

def build_vocab(data, counter, threshold = 3):
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
    
def load_data(data_path = '../data/training/training-data.1m', ws=1):
    
    train_data = read_file(data_path)
    train_data = preprocess(train_data)
    vocab_to_idx, idx_to_vocab = build_vocab(train_data)
    train_data = vectorize(train_data, vocab_to_idx, ws)
    np.save('idx_to_vocab', idx_to_vocab)
    np.save('train_data', train_data)
    return train_data, idx_to_vocab

def task(idx, q, word_set, data_path):
    data = read_file(data_path)
    data = preprocess(data)
    for sentence in data:
            for word in sentence:
                if word in word_set:
                    q.put(sentence)
                    break
    time.sleep(0.1)
    print('Process {} finished'.format(idx))
    return

if __name__ == "__main__":
    
    dev_word_set = load_csv('../data/similarity/dev_x.csv')
    test_word_set = load_csv('../data/similarity/test_x.csv')
    word_set = dev_word_set | test_word_set
    #word_set = {WordNetLemmatizer().lemmatize(w) for w in word_set}

    train_data = []
    q = multiprocessing.Queue()
    processes = []
    path = '../data/1-billion-word-language-modeling-benchmark-r13output/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/'
    for i in range(1, 100):
        
        data_path = path + 'news.en-000{:02d}-of-00100'.format(i)
        p = multiprocessing.Process(target = task, args=(i, q, word_set,data_path))
        processes.append(p)
        p.start()
    counter = {word:0 for word in word_set}
    while True:
        item = q.get()
        if q.qsize() == 0:
            break
        else:
            for word in item:
                if word in counter and counter[word] < 2300:
                    train_data.append(item)
                    counter[word] += 1
                    break
    print(len(train_data))
    for process in processes:
        process.join()
    # counter = {}
    # while True:
    #     item = q.get()
    #     if q.qsize() == 0:
    #         break
    #     else:
    #         for word in item:
    #             if word not in counter:
    #                 counter[word] = 1
    #             else:
    #                 counter[word] += 1

    # for process in processes:
    #     process.join()

    counter = count_vocab(train_data)
    vocab_to_idx, idx_to_vocab = build_vocab(train_data, counter)
    train_data = vectorize(train_data, vocab_to_idx, 5)
    #np.save('counter', counter)
    np.save('idx_to_vocab_400_ds_5', idx_to_vocab)
    np.save('train_data_400_ds_5', train_data)
