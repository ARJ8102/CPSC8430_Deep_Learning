# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# import libraries
import sys
import torch
import json
from torch.utils.data import DataLoader
from bleu_eval import BLEU
import main6
#from Main import test_data, test, MODELS, encoderRNN, decoderRNN, attention
import pickle

if not torch.cuda.is_available():
    modelIP = torch.load('Deep Learning CPSC 8430/HW_2/hw2/SavvedModels/model0(3).h5', map_location=lambda storage, loc: storage)
else:
    modelIP = torch.load('SavvedModels/model0(3).h5')


files_dir = '/home/atharvj/Deep Learning CPSC 8430/HW_2/hw2/testing_data/feat'
label_files = '/home/atharvj/Deep Learning CPSC 8430/HW_2/hw2/testing_label.json'
word_min = 3
i2w,w2i,dictonary = main6.dictonaryFunc(word_min)

test_dataset = main6.test_dataloader(files_dir)
test_dataloader = main6.DataLoader(dataset = test_dataset, batch_size=1, shuffle=True, num_workers=8)

model = modelIP.cuda()


# +
ss = main6.test(test_dataloader, model, i2w)

with open('test_output.txt', 'w') as f:
    for id, s in ss:
        f.write('{},{}\n'.format(id, s))


# +
# Bleu Eval
test = json.load(open('testing_label.json','r'))
#output = 'testing_data.txt'
output = 'test_output.txt'
result = {}

with open(output,'r') as f:
    for line in f:
        line = line.rstrip()
        comma = line.index(',')
        test_id = line[:comma]
        caption = line[comma+1:]
        result[test_id] = caption
#count by the method described in the paper https://aclanthology.info/pdf/P/P02/P02-1040.pdf
bleu=[]
for item in test:
    score_per_video = []
    captions = [x.rstrip('.') for x in item['caption']]
    score_per_video.append(BLEU(result[item['id']],captions,True))
    bleu.append(score_per_video[0])
average = sum(bleu) / len(bleu)
print("Average bleu score is " + str(average))
# -


