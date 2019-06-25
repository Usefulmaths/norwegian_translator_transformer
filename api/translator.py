import torch
import torch.nn as nn
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from language import Preprocessor, Language
from model import Transformer


class Translator(object):
    def __init__(self, english_language, norwegian_language, cuda='cpu'):
        self.cuda = cuda
        self.model = self.instantiate_model(english_language.vocab_size(),
                                            norwegian_language.vocab_size())

        self.preprocessor = Preprocessor()
        self.english_language = english_language
        self.norwegian_language = norwegian_language

        self.bleu1 = -1

    def instantiate_model(self, english_vocab_size, norwegian_vocab_size, embedding_dim=256, num_heads=8, num_encoders=6, ff_dim=256):
        model = Transformer(english_vocab_size,
                            norwegian_vocab_size,
                            embedding_dim,
                            num_heads,
                            num_encoders,
                            ff_dim, self.cuda).to(self.cuda)

        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform(p)

        return model

    def train(self, X_train, y_train, X_test=None, y_test=None, epochs=500, train_batch_size=128, test_batch_size=128, log_batch=20, early_stopping_criteria=200, file_save_name=None, file_load_name=None):
        if file_load_name is not None:
            self.load_model(file_load_name)

        X_train = torch.tensor(X_train).to(self.cuda)
        y_train = torch.tensor(y_train).to(self.cuda)

        train_dataset = SentenceDataSet(X_train, y_train)
        train_dataloader = DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True)

        if X_test is not None:
            X_test = torch.tensor(X_test).to(self.cuda)
            y_test = torch.tensor(y_test).to(self.cuda)

            test_dataset = SentenceDataSet(X_test, y_test)
            test_dataloader = DataLoader(
                test_dataset, batch_size=test_batch_size, shuffle=True)

        optim = torch.optim.Adam(self.model.parameters(), lr=0.0002)
        cross_entropy = nn.CrossEntropyLoss()

        early_stopping = 0

        for epoch in range(epochs):
            if early_stopping >= early_stopping_criteria:
                break

            for batch_idx, (x, y) in enumerate(train_dataloader):
                self.model.train()
                if early_stopping >= early_stopping_criteria:
                    break

                x = x.to(self.cuda)
                y = y.to(self.cuda)

                output_sentence = self.model(x, y).to(
                    self.cuda)[:, :-1]  # predictions, beyond <sos>
                target_sentence = y[:, 1:]

                total_loss = torch.zeros(
                    (train_batch_size, output_sentence.shape[1]))

                for i in range(output_sentence.shape[1]):

                    loss = cross_entropy(
                        output_sentence[:, i], target_sentence[:, i])

                    total_loss[:, i] = loss

                total_loss = total_loss.mean()

                optim.zero_grad()
                total_loss.backward()

                torch.nn.utils.clip_grad_norm(self.model.parameters(), 3.0)
                optim.step()

                if batch_idx % log_batch == 0:
                    for batch_idy, (x, y) in enumerate(test_dataloader):
                        self.model.eval()
                        x = x.to(self.cuda)
                        y = y.to(self.cuda)

                        output_sentence = self.model(
                            x, y)[:, :-1].to(self.cuda)
                        target_sentence = y[:, 1:]

                        total_test_loss = torch.zeros(
                            (test_batch_size, output_sentence.shape[1]))

                        if x.shape[0] == test_batch_size:
                            for i in range(output_sentence.shape[1]):
                                loss = cross_entropy(
                                    output_sentence[:, i], target_sentence[:, i])
                                total_test_loss[:, i] = loss

                            total_test_loss = total_test_loss.mean()

                            preds = torch.argmax(output_sentence, dim=2)

                            bleu_scores = self.bleu_score_batch(
                                target_sentence, preds)

                            bleu1 = bleu_scores[0]

                            if self.model.bleu1 == -1:
                                self.model.bleu1 = bleu1

                            if bleu1 < self.model.bleu1:
                                early_stopping += 1

                            else:
                                self.model.bleu1 = bleu1
                                early_stopping = 0

                                if file_save_name is not None:
                                    torch.save(self.model.state_dict(), '%s' %
                                               file_save_name)

                        break
                    input_sentence = '<sos> i dont like your house <eos>'
                    input_sentence = padding(input_sentence, 20)

                    output_sentence = '<sos>'
                    output_sentence = padding(output_sentence, 20)

                    translation = translator.translate(
                        input_sentence, output_sentence)
                    print(translation)
                    print("Epoch %d, Batch %d -> Train Loss: %.4f\tTest Loss: %.4f\tBleu-1: %.4f (ES: %d)\tBleu-2: %.4f\tBleu-3: %.4f\tBleu-4: %.4f" %
                          (epoch, batch_idx, total_loss, total_test_loss, bleu1, early_stopping, bleu_scores[1], bleu_scores[2], bleu_scores[3]))

    def load_model(self, file_name):
        if self.cuda == 'cpu':
            self.model.load_state_dict(
                torch.load(file_name, map_location='cpu'))

        else:
            self.model.load_state_dict(torch.load(file_name))

    def translate(self, input_sentence, output_sentence='<sos>', padding=30):
        input_sentence = self.add_padding(input_sentence, padding)
        output_sentence = self.add_padding(output_sentence, padding)

        # Convert to indices
        input_indices = self.english_language.sentence_to_idx(input_sentence)
        output_indices = self.norwegian_language.sentence_to_idx(
            output_sentence)

        input_indices = torch.tensor(input_indices).view(
            1, len(input_indices)).to(self.cuda)
        output_indices = torch.tensor(output_indices).view(
            1, len(output_indices)).to(self.cuda)

        # Norwegian indices prediction
        prediction = self.model.predict(input_indices, output_indices)
        prediction = prediction.data.cpu().numpy()[0].tolist()

        # Convert Norwegian indices to sentence
        translation = self.norwegian_language.idx_to_sentence(prediction)

        # Remove <sos> and <eos>
        translation = [word for word in translation if word != '<sos>']
        translation = [word for word in translation if word != '<eos>']

        translation = ' '.join(translation).replace('<pad>', '')

        return translation

    def add_padding(self, sentence, max_length):
        words = sentence.split(' ')
        length = len(words)

        for i in range(max_length - length):
            words.append('<pad>')

        return ' '.join(words)

    def get_attention_matrix(self, sentence):
        # Preprocess
        preprocessor = Preprocessor()
        sentence = preprocessor.preprocess_sentence(sentence)

        # Convert to indices
        indices = self.english_language.sentence_to_idx(sentence)

        indices = torch.tensor(
            indices).view(1, len(indices))

        attention_matrix = self.model.get_attention_matrix(
            indices.to(self.cuda))

        return attention_matrix

    def bleu_score_batch(self, target_sents, predicted_sents):
        batch_size = len(target_sents)

        bleus = np.zeros((batch_size, 4))

        for i in range(len(target_sents)):
            bleu1, bleu2, bleu3, bleu4 = self.bleu_score(
                target_sents[i], predicted_sents[i])
            bleus[i][0] = bleu1
            bleus[i][1] = bleu2
            bleus[i][2] = bleu3
            bleus[i][3] = bleu4

        avg_bleus = np.mean(bleus, axis=0)
        return avg_bleus

    def bleu_score(self, target_sent, predicted_sent):
        if type(target_sent) is torch.Tensor:
            target_sent = target_sent.data.cpu().numpy()
            predicted_sent = predicted_sent.data.cpu().numpy()

            # Remove sos(0), eos(1), and padding(3)
            target_sent = [idx for idx in target_sent if idx !=
                           0 and idx != 1 and idx != 3]
            predicted_sent = [
                idx for idx in predicted_sent if idx != 0 and idx != 1 and idx != 3]

        else:
            target_sent = [word for word in target_sent if word !=
                           '<sos>' and word != '<eos>' and word != '<pad>']
            predicted_sent = [
                word for word in predicted_sent if word != '<sos>' and word != '<eos>' and word != '<pad>']

        if len(predicted_sent) == 0:
            return 0, 0, 0, 0

        smoothie = SmoothingFunction().method1

        bleu1 = sentence_bleu(
            [target_sent], predicted_sent, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu2 = sentence_bleu(
            [target_sent], predicted_sent, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        bleu3 = sentence_bleu([target_sent], predicted_sent,
                              weights=(1./3, 1./3, 1./3, 0), smoothing_function=smoothie)
        bleu4 = sentence_bleu([target_sent], predicted_sent,
                              weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

        return bleu1, bleu2, bleu3, bleu4
