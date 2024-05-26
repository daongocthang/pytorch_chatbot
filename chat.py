import os
import random
from brain import Brain
from utils import load_yaml, __basedir

brain = Brain()

intents = load_yaml(os.path.join(__basedir, 'intents.yaml'))['intents']

print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input('You: ')
    if sentence.lower() == 'quit':
        break
    prob, tag = brain.think(sentence)
    print("prob:", prob.item(), "tag:", tag)
    if prob.item() > 0.9:
        for intent in intents:
            if tag == intent['tag']:
                print(f'Bot:', random.choice(intent['responses']))
                break
    else:
        print(f'Bot: Tôi không hiểu...')
