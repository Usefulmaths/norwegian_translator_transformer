import pickle as pkl
import torch
from language import Language
from translator import Translator

if __name__ == '__main__':
    english_language = pkl.load(
        open('./data/languages/english_language', 'rb'))
    norwegian_language = pkl.load(
        open('./data/languages/norwegian_language', 'rb'))

    file_load_name = './data/model/translator_model_transformer.py'
    translator = Translator(english_language, norwegian_language, cuda='cpu')
    translator.load_model(file_load_name)

    print(translator.translate('do you have a friend'))
