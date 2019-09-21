import numpy as np
import re
from language import Language, Preprocessor
import pickle as pkl


def retrieve_raw_data(root_dir, translation_file, number_of_lines=20000000):
    english_sentences = []
    norwegian_sentences = []

    sentence_dict = dict()

    with open(root_dir + translation_file, 'r') as f:
        lines = f.readlines()

        for line in lines[:number_of_lines]:
            if 'tuv xml:lang="en"' in line:
                english_sentence = re.search(
                    '<seg>(.*)</seg>', line).group(1)

                english_sentences.append(english_sentence)

            if 'tuv xml:lang="no"' in line:
                norwegian_sentence = re.search(
                    '<seg>(.*)</seg>', line).group(1)

                norwegian_sentences.append(norwegian_sentence)

    return english_sentences, norwegian_sentences


def preprocess_data(english_sentences, norwegian_sentences):
    preprocessor = Preprocessor()

    preprocessed_english_sentences = []
    preprocessed_norwegian_sentences = []

    for sentence in english_sentences:
        pre_eng = preprocessor.preprocess_sentence(sentence)
        preprocessed_english_sentences.append(pre_eng)

    for sentence in norwegian_sentences:
        pre_nor = preprocessor.preprocess_sentence(
            sentence, sos=True, eos=True)
        preprocessed_norwegian_sentences.append(pre_nor)

    return preprocessed_english_sentences, preprocessed_norwegian_sentences


def retrieve_data(root_dir, translation_file):
    # Retrieve
    english_sentences, norwegian_sentences = retrieve_raw_data(
        root_dir, translation_file)

    # Preprocess
    english_sentences, norwegian_sentences = preprocess_data(
        english_sentences, norwegian_sentences)

    return english_sentences, norwegian_sentences


def indices_from_sentence(language, sentence):
    indices = []

    for word in sentence.split(' '):
        try:
            idx = language.word2idx[word]
        except(KeyError):
            idx = language.word2idx['<unk>']

        indices.append(idx)

    return indices


def sentence_from_indices(language, indices):
    sentence = []

    for index in indices:
        word = language.idx2word[index]
        sentence.append(word)

    return sentence


def convert_sentences_index(language, sentences):
    sentences_indices = []
    for sentence in sentences:
        indices = indices_from_sentence(language, sentence)
        sentences_indices.append(indices)

    sentences_indices = np.array(sentences_indices)

    return sentences_indices


def filter_sentences_by_length(english_sentences, norwegian_sentences, max_length):
    filtered_english = []
    filtered_norwegian = []

    for i in range(len(english_sentences)):
        if len(english_sentences[i].split(' ')) < max_length and len(norwegian_sentences[i].split(' ')) < max_length:
            filtered_english.append(english_sentences[i])
            filtered_norwegian.append(norwegian_sentences[i])

    return filtered_english, filtered_norwegian


def pad_sentences(sentences):
    max_length = max([len(sentence.split(' ')) for sentence in sentences])
    padded_sentences = []

    for sentence in sentences:
        words = sentence.split(' ').copy()

        while(len(words) < max_length):
            words.append('<pad>')

        words = ' '.join(words)
        padded_sentences.append(words)

    return padded_sentences


def filter_pad_sentences(english_sentences, norwegian_sentences, max_length):
    filtered_english, filtered_norwegian = filter_sentences_by_length(
        english_sentences, norwegian_sentences, max_length)

    padded_english_sentences = pad_sentences(filtered_english)
    padded_norwegian_sentences = pad_sentences(filtered_norwegian)

    return padded_english_sentences, padded_norwegian_sentences


def train_test_sentences(filtered_english, filtered_norwegian, test_size=0.01):
    number_of_sentences = len(filtered_english)
    indices = np.arange(number_of_sentences)

    np.random.shuffle(indices)

    train_indices = indices[:int(number_of_sentences * (1 - test_size))]
    test_indices = indices[int(number_of_sentences * (1 - test_size)):]

    english_train_sentences = []
    norwegian_train_sentences = []

    english_test_sentences = []
    norwegian_test_sentences = []

    for index in train_indices:
        english_train_sentence = filtered_english[index]
        norwegian_train_sentence = filtered_norwegian[index]

        english_train_sentences.append(english_train_sentence)
        norwegian_train_sentences.append(norwegian_train_sentence)

    for index in test_indices:
        english_test_sentence = filtered_english[index]
        norwegian_test_sentence = filtered_norwegian[index]

        english_test_sentences.append(english_test_sentence)
        norwegian_test_sentences.append(norwegian_test_sentence)

    return english_train_sentences, norwegian_train_sentences, english_test_sentences, norwegian_test_sentences
