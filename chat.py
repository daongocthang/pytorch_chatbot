import os
import random
from prediction import predict
from utils import load_yml, __basedir

intents = load_yml(os.path.join(__basedir, 'intents.yml'))['intents']

print("Let's chat! (type '/stop' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input('You: ')
    if sentence.lower() == '/stop':
        break
    prob, tag = predict(sentence)
    print("prob:", prob.item(), "tag:", tag)
    if prob.item() > 0.9:
        for intent in intents:
            if tag == intent['tag']:
                print(f'Bot:', random.choice(intent['responses']))
                break
    else:
        print(f'Bot: Tôi không hiểu...')
