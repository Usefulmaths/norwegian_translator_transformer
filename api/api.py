import flask
from flask import request, jsonify
import pickle as pkl
import torch
from language import Language
from translator import Translator


app = flask.Flask(__name__)
app.config['DEBUG'] = True

english_language = pkl.load(
    open('./data/languages/english_language', 'rb'))

norwegian_language = pkl.load(
    open('./data/languages/norwegian_language', 'rb'))

file_load_name = '../data/model/translator_model_transformer.py'
translator = Translator(english_language, norwegian_language, cuda='cpu')
translator.load_model(file_load_name)


@app.route('/translate', methods=['GET'])
def translate():

    english = request.args['english']
    norwegian = translator.translate(english)

    return "<h1>%s</h1><h1>%s</h1>" % (english, norwegian)


app.run()
