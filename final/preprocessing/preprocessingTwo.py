#!/usr/bin/env python3

'''

Preprocessing

'''

import logging
import re
import itertools
import json
from autocorrect import spell
from os.path import abspath, exists
import csv
import sys
import numpy as np
import pandas as pd
from wordsegment import load, segment
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from html.parser import HTMLParser
logging.basicConfig(level=logging.INFO)


def preprocess(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    y = []
    corpus = []
    load()
    stopWords = set(stopwords.words('english'))
    data = {'index Label' :[], 'Tweet Text': []};

    with open(fp, 'rt', encoding='utf-8') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                label = int(line.split("\t")[1])
                tweet = line.split("\t")[2]

                 # remove url
                tweet = re.sub(r'(http|https?|ftp)://[^\s/$.?#].[^\s]*', '', tweet, flags=re.MULTILINE)
                tweet = re.sub(r'[http?|https?]:\\/\\/[^\s/$.?#].[^\s]*', '', tweet, flags=re.MULTILINE)

                # remove mentions
                remove_mentions = re.compile(r'(?:@[\w_]+)') 
                tweet = remove_mentions.sub('',tweet)

                # remove emoticons
                try:
                    emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F" # emoticons
                        u"\U0001F300-\U0001F5FF" # symbols & pictographs
                        u"\U0001F680-\U0001F6FF" # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF" # flags (iOS)
                        "]+", flags=re.UNICODE)
                    tweet = emoji_pattern.sub(r'', tweet)
                except Exception:
                    pass


                # remove unicode
                try:
                    tweet = tweet.encode('ascii').decode('unicode-escape').encode('ascii','ignore').decode("utf-8")
                except Exception: 
                    pass

                # remove more unicode characters 
                try:
                    tweet = tweet.encode('ascii').decode('unicode-escape')
                    tweet = HTMLParser().unescape(tweet)
                except Exception:
                    pass

                # Standardising words
                tweet = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet))

                # contractions applied
                words = tweet.split()
                tweet = [apoDict[word] if word in apoDict else word for word in words]
                tweet = " ".join(tweet)

                hashWords =  re.findall(r'(?:^|\s)#{1}(\w+)', tweet)
                # replace #word with word
                tweet = re.sub(r'(?:^|\s)#{1}(\w+)', r' \1', tweet)

                # word segmentation
                token_list =  word_tokenize(tweet)
                segmented_word = []
                for i in token_list:
                    if i in hashWords:
                        seg = segment(i)
                        segmented_word.extend(seg)
                    else:
                        segmented_word.append(i)

                tweet = ' '.join(segmented_word)

                # remove special symbols
                tweet = re.sub('[@#$._|]', ' ', tweet)

                # remove extra whitespaces
                tweet = re.sub('[\s]+', ' ', tweet)

                # Spelling correction
                spell_list = word_tokenize(tweet)
                filterlist = []
                for i in spell_list:
                    correct_spell = spell(i)
                    filterlist.append(correct_spell)
                tweet = ' '.join(filterlist)

                # lemma
                tweet = word_tokenize(tweet)
                lemma_tweet = []
                for tweet_word in tweet:
                    lemma_tweet.append(WordNetLemmatizer().lemmatize(tweet_word,'v'))

                tweet = ' '.join(lemma_tweet)

                # remove stop words
                token_list = word_tokenize(tweet)
                wordsFiltered = []
                for i in token_list:
                    if i not in stopWords:
                        wordsFiltered.append(i)
                tweet = ' '.join(wordsFiltered)

                # remove open or empty lines
                if not re.match(r'^\s*$', tweet):
                    if not len(tweet) <= 3:
                        corpus.append(tweet)
                        y.append(label)
                        print(tweet)

                #creating a dictionary
                data['Tweet Text'].append(tweet)
                data['index Label'].append(label)
    #creating a dataframe
    df = pd.DataFrame(data)
    #to order the columns in the csv file (while copying from dataframes to a CSV file)
    df_reorder = df[['index Label', 'Tweet Text']]
    #writing dataframe to a csv file
    #df_reorder.to_csv('semeval_preprocessed_training_data.csv', encoding='utf-8', index= False)
    df_reorder.to_csv('semeval_preprocessed_test_data.csv', encoding='utf-8', index= False)

    return corpus, y

# Contractions
def loadAppostophesDict(fp):
    apoDict = json.load(open(fp))
    return apoDict

if __name__ == "__main__":
    DATASET_FP = "./SemEval2018-T3_gold_test_taskA_emoji.txt"
    #DATASET_FP = "./SemEval2018-T3-train-taskA.txt"
    APPOSTOPHES_FP = "./appos.txt"
    apoDict = loadAppostophesDict(APPOSTOPHES_FP)
    corpus, y = preprocess(DATASET_FP)
