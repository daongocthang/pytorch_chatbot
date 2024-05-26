import os

import torch
from model import NeuralNet
from nltk_vi import tokenize, bag_of_words

basedir = os.path.dirname(os.path.abspath(__file__))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FILE = 'data/data.pth'


class Brain:
    def __init__(self):
        print('Loading data...')
        data = torch.load(os.path.join(basedir, FILE))

        input_size = data['input_size']
        hidden_size = data['hidden_size']
        output_size = data['output_size']
        all_words = data['all_words']
        tags = data['tags']
        model_state = data['model_state']

        model = NeuralNet(
            input_size,
            hidden_size,
            output_size
        ).to(device)
        model.load_state_dict(model_state)
        model.eval()

        self.model = model
        self.all_words = all_words
        self.tags = tags

    def think(self, sentence):
        sentence = tokenize(sentence)
        x = bag_of_words(sentence, self.all_words)
        x = x.reshape(1, x.shape[0])
        x = torch.from_numpy(x).to(device)

        output = self.model(x)
        _, predicted = torch.max(output, dim=1)

        tag = self.tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        return prob, tag
