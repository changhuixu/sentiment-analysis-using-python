# This script is what created the dataset pickled.

dataset_path='/space/changhxu/LDA/project/aclImdb/'

import numpy
import cPickle as pkl

from collections import OrderedDict
from nltk.corpus import stopwords

import glob
import os
import re
import string

def extract_words(sentences):
    result = []
    stop = stopwords.words('english')
    trash_characters = '?.,!:;"$%^&*()#@+/0123456789<>=\\[]_~{}|`'
    trans = string.maketrans(trash_characters, ' '*len(trash_characters))

    for text in sentences:
        text = re.sub(r'[^\x00-\x7F]+',' ', text)
        text = text.replace('<br />', ' ')
        text = text.replace('--', ' ').replace('\'s', '')
        text = text.translate(trans)
        text = ' '.join([w for w in text.split() if w not in stop])

        words = []
        for word in text.split():
            word = word.lstrip('-\'\"').rstrip('-\'\"')
            if len(word)>2:
                words.append(word.lower())
        text = ' '.join(words)
        result.append(text.strip())
    return result


def grab_data(path):
    sentences = []
    currdir = os.getcwd()
    os.chdir(path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir(currdir)
    sentences = extract_words(sentences)

    return sentences

def main():
    # Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/
    path = dataset_path

    train_x_pos = grab_data(path+'train/pos')
    train_x_neg = grab_data(path+'train/neg')
    train_x = train_x_pos + train_x_neg
    train_y = [1] * len(train_x_pos) + [0] * len(train_x_neg)

    test_x_pos = grab_data(path+'test/pos')
    test_x_neg = grab_data(path+'test/neg')
    test_x = test_x_pos + test_x_neg
    test_y = [1] * len(test_x_pos) + [0] * len(test_x_neg)

    f = open('train.pkl', 'wb')
    pkl.dump((train_x, train_y), f, -1)
    f.close()
    f = open('test.pkl', 'wb')
    pkl.dump((test_x, test_y), f, -1)
    f.close()


if __name__ == '__main__':
    main()
