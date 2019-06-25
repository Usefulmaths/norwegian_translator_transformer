

from collections import defaultdict
import operator
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import torch.nn.functional as F

from torchnlp.metrics import bleu

from sklearn.model_selection import train_test_split

import re
import string
import unicodedata
import nltk
import contractions
import inflect


class Preprocessor(object):
    def remove_non_ascii(self, words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode(
                'ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words

    def to_lowercase(self, words):
        """Convert all characters to lowercase from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words

    def remove_punctuation(self, words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)

            if new_word != '':
                new_words.append(new_word)

        return new_words

    def split_punctuation(self, words):
        new_words = []

        for word in words:
            if word[-1] == '.' or word[-1] == '?' or word[-1] == '!':
                punc = word[-1]
                word = word[:-1]

                new_words.append(word)
                new_words.append(punc)

            else:
                new_words.append(word)
        return new_words

    def replace_numbers(self, words):
        """Replace all interger occurrences in list of tokenized words with textual representation"""
        p = inflect.engine()
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = p.number_to_words(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words

    def add_eos(self, words):
        words_copy = words.copy()
        words_copy.append('<eos>')

        return words_copy

    def add_sos(self, words):
        words_copy = words.copy()
        words_copy.insert(0, '<sos>')

        return words_copy

    def preprocess_sentence(self, sentence, eos=False, sos=False):
        sentence_array = sentence.split(' ')
        sentence_array = self.to_lowercase(sentence_array)
        sentence_array = self.remove_punctuation(sentence_array)
        sentence_array = self.split_punctuation(sentence_array)
        sentence_array = self.replace_numbers(sentence_array)

        if eos:
            sentence_array = self.add_eos(sentence_array)

        if sos:
            sentence_array = self.add_sos(sentence_array)

        sentence = ' '.join(sentence_array)

        return sentence


class Language(object):
    def __init__(self, name):
        self.name = name
        self.word2idx = {'<sos>': 0, '<eos>': 1, '<unk>': 2, '<pad>': 3}
        self.idx2word = {0: '<sos>', 1: '<eos>', 2: '<unk>', 3: '<pad>'}
        self.word_counter = 4

        self.word_frequency = defaultdict(int)

    def sentence_to_idx(self, sentence):
        indices = []
        for word in sentence.split(' '):
            try:
                idx = self.word2idx[word]

            except(KeyError):
                idx = self.word2idx['<unk>']

            indices.append(idx)

        return indices

    def idx_to_sentence(self, indices):
        sentence = []

        for index in indices:
            word = self.idx2word[index]
            sentence.append(word)

        return sentence

    def index_words(self):
        for word in self.top_words:
            if word not in self.word2idx.keys():
                self.word2idx[word] = self.word_counter
                self.idx2word[self.word_counter] = word
                self.word_counter += 1

    def count_words(self, sentence):
        for word in sentence.split(' '):
            self.count_word(word)

    def count_word(self, word):
        self.word_frequency[word] += 1

    def top_n_words(self, n):
        sorted_freq_words = list(
            sorted(self.word_frequency.items(), key=lambda kv: kv[1], reverse=True))[:n]
        sorted_words = list(map(lambda x: x[0], sorted_freq_words))
        self.top_words = sorted_words

        return sorted_words

    def vocab_size(self):
        return len(list(self.word2idx.keys()))
