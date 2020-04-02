from argparse import ArgumentParser
import sys

import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from scipy import spatial
#from sklearn.metrics.pairwise import cosine_similarity

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def read_embedding(path):

    embedding = {}
    dim = None

    for row in open(path):

        word, *vector = row.split()
        embedding[word] = [float(x) for x in vector]

        if dim and len(vector) != dim:

            print("Inconsistent embedding dimensions!", file = sys.stderr)
            sys.exit(1)

        dim = len(vector)

    return embedding, dim


parser = ArgumentParser()

parser.add_argument("-e", "--embedding", dest = "emb_path",
    required = True, help = "path to your embedding")

parser.add_argument("-w", "--words", dest = "pairs_path",
    required = True, help = "path to dev_x or test_x word pairs")

args = parser.parse_args()


E, dim = read_embedding(args.emb_path)
pairs = pd.read_csv(args.pairs_path, index_col = "id")
def score(v1, v2):
    return np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))

# lemmatizer.lemmatize(w, get_wordnet_pos(w))
lemmatizer = WordNetLemmatizer()
res = []
for w1, w2 in zip(pairs.word1, pairs.word2):
    #w1 = lemmatizer.lemmatize(w1.lower(), get_wordnet_pos(w1.lower()))
    #w2 = lemmatizer.lemmatize(w2.lower(), get_wordnet_pos(w2.lower()))
    w1, w2 = w1.lower(), w2.lower()
    if w1 not in E:
        w1_vec = E['UNK']
    else:
        w1_vec = E[w1]
    if w2 not in E:
        w2_vec = E['UNK']
    else:
        w2_vec = E[w2]
    res.append(np.dot(w1_vec, w2_vec))
    #w1_vec = np.array(w1_vec).reshape(-1, 1)
    #w2_vec = np.array(w2_vec).reshape(-1, 1)
    #cos = np.dot(w1_vec, w2_vec) / (np.linalg.norm(w1_vec) * np.linalg.norm(w2_vec))
    #res.append(cos)
    # res.append(1 - spatial.distance.cosine(w1_vec, w2_vec))
pairs["similarity"] = res
# pairs["similarity"] = [np.dot(E[lemmatizer.lemmatize(w1.lower(), get_wordnet_pos(w1.lower()))], E[lemmatizer.lemmatize(w2.lower(), get_wordnet_pos(w2.lower()))]) if lemmatizer.lemmatize(w1.lower(), get_wordnet_pos(w1.lower())) in E and lemmatizer.lemmatize(w2.lower(), get_wordnet_pos(w2.lower())) in E else 0
#     for w1, w2 in zip(pairs.word1, pairs.word2)]

del pairs["word1"], pairs["word2"]
print("Detected a", dim, "dimension embedding.", file = sys.stderr)
pairs.to_csv(sys.stdout)
