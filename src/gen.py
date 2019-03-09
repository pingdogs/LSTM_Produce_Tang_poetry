import argparse
import pickle
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

model_path = './output/model/ep30/model.bin'
corpus = 'corpus.txt'
def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
def load_rhyme(path):
    with open(path) as f:
        data = json.load(f)
    return data
def is_end(c):
    end_tokens = ['。', '？', '！', '.', '?', '!']
    return c in end_tokens

def to_prob(vec):
    s = sum(vec)
    return [v / s for v in vec]

def gen_text(model, patterns, char_to_int, int_to_char, chars, n_sent=10, restart_seq=False, ty = 5):

    n_patterns = len(patterns)

    # Randomly choose a pattern to start text generation
    start = np.random.randint(0, n_patterns - 1)
    pattern = patterns[start]

    # Start generation until n_sent sentences generated 
    cnt = 0
    rhyindex = -1
    temp = ""
    result = ""
    while cnt < n_sent: 
        seq_in = np.array(pattern)
        seq_in = seq_in.reshape(1, -1) # batch_size = 1

        seq_in = Variable(torch.LongTensor(seq_in))

        # Predict next character
        pred = model(seq_in)
        # pred = to_prob(F.softmax(pred, dim=1).data[0].numpy()) # turn into probability distribution
        tt = F.softmax(pred, dim=1)
        # print(tt.data.cpu().numpy())

        if len(temp) == ty-1 and cnt % 2 == 1:
            if cnt == 1:
                char_idx = tt.data.max(dim=1,keepdim=True)[1]
                char = int_to_char[char_idx.data.cpu().numpy()[0][0]]
                for a in range(0, len(rhymelist)):
                    if char in rhymelist[a]:
                        rhyindex = a
                        break
                if rhyindex == -1:
                    start = np.random.randint(0, n_patterns - 1)
                    pattern = patterns[start]
                    temp = ""
                    continue
            else:
                lst = rhymelist[rhyindex]
                clean_space = lst.split(" ")
                maxIndex = -1
                probilities = tt.data.cpu().numpy()
                for word in clean_space:
                    if word not in char_to_int:
                        continue
                    c = char_to_int[word]
                    pob = probilities[0][c]
                    if maxIndex == -1:
                        maxIndex = c
                    elif pob > probilities[0][maxIndex] and int_to_char[c] not in result:
                        maxIndex = c
                char = int_to_char[maxIndex]
        else:
            char_idx = tt.data.max(dim=1,keepdim=True)[1]
            char = int_to_char[char_idx.data.cpu().numpy()[0][0]]

        

        if not is_end(char):
            if char not in result and char not in temp:
                temp += char
            else:
                start = np.random.randint(0, n_patterns - 1)
                pattern = patterns[start]
                temp = ""
                continue

        pattern.append(char_idx)
        pattern = pattern[1:]

        if is_end(char):
            if len(temp) == ty:
                result += temp + "。"
                cnt += 1 
            temp = ""
            if restart_seq:
                start = np.random.randint(0, n_patterns - 1)
                pattern = patterns[start]
                print()
    print(result)
    if not restart_seq:
        print()

rhymelist = load_rhyme("rhyme.json")["rhyme"]

if __name__ == '__main__':

    dataX, char_to_int, int_to_char, chars = load_pickle(corpus)

    # Load model
    model = torch.load(model_path, map_location='cpu')
    
    # Generate text
    gen_text(model, dataX, char_to_int, int_to_char, chars, n_sent=8, restart_seq=False, ty = 5)