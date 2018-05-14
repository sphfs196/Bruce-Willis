# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import time
import math
import csv


# time calculating function

def time_since(since):
	s = time.time() - since
	m = math.floor(s/60)
	s -= m*60
	return '%dm %ds' % (m, s)



# initialize

CONTEXT_SIZE = 2  # input character length
EMBEDDING_DIM = 10  # word vector length
EPOCH_TO_RUN = 10

filePath = ""
fileName = "newsCorpus.txt"
vocabFilePath = 'news_vocab.csv'
modelPath_saveTo = "10_front_model.pkl" # path to save
argPath_loadFrom = "10_front_arg.pt" # path to load from
argPath_saveTo = "10_front_arg.pt" # path to save


# preparing training data
# read in the news corpus file to generate training data
# data form: (['A', 'B'], 'C')

newsList = []
with open(filePath + fileName, 'r', encoding='utf-8') as newsCorpus:
    for news_idx, news_text in enumerate(newsCorpus):
        newsList.append(news_text[:len(news_text)-1])

trigrams = []
trigramsList = []
for news in newsList:
    trigrams = [ ([news[i+1], news[i+2]], news[i])
            for i in range(len(news) - 2) ]
    trigramsList.append(trigrams)


# set vocab
# read in the vocabulary set constructed from the news corpus
# to make sure that the index won't change in each time training
# create a index <-> word dictionary

ix_to_word = {}
with open(vocabFilePath, mode='r', encoding='utf-8') as f:
    reader = csv.reader(f)
    ix_to_word = {rows[0]:rows[1] for rows in reader}


del[ix_to_word['Ind']]
ix_to_word = {int(k):v for k,v in ix_to_word.items()}
word_to_ix = {v:k for k,v in ix_to_word.items()}


"""
vocab = set()
for news in newsList:
    vocab.update(set(news))

word_to_ix = {word : i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}
"""


# define model class

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embdeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embdeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs, self.embeddings




# setting model, optimizer

print('loading model...')
model = NGramLanguageModeler(len(ix_to_word), EMBEDDING_DIM, CONTEXT_SIZE)
losses = []
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, eps=0.0001, betas=(0.99, 0.999))
start_epoch = 0



# UNCOMMENT IF LOADING ARGS !!!!!!!!!!!!!!!
"""
# loading pretrained model, optimizer settings...

checkpoint = torch.load(argPath_loadFrom)
start_epoch = checkpoint['epoch']
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
print("=> loaded checkpoint : (start_epoch {})".format(checkpoint['epoch']))
"""



# cpu / gpu

if torch.cuda.is_available():
    print("in gpu")
    model.cuda()
else:
    print("in cpu")
    model.cpu()



# training

start_time = time.time()
print('start training...')

for epoch in range(start_epoch, start_epoch + EPOCH_TO_RUN):
    total_loss = torch.cuda.FloatTensor([0])
    
    for trigramsItem in trigramsList:
        for context, target in trigramsItem:
            # �Ncontext(ex : ['When','forty'])�ഫ��index(ex : [68, 15])
            # �A�নpytorch��variable
            context_idxs = [word_to_ix[w] for w in context]
            context_var = autograd.Variable(torch.cuda.LongTensor(context_idxs))

            # �M��gradient�A����W�@�����֭p
            model.zero_grad()

            # ��variable�ܼƶi�h�]forward
            log_probs, embedd = model(context_var)

            # �p��loss(��target variable��i�h)
            loss = loss_function(log_probs, autograd.Variable(torch.cuda.LongTensor([word_to_ix[target]])))

            # �]backward�A��sgradient
            loss.backward()
            optimizer.step()

            total_loss += loss.data
        losses.append(total_loss)
    
    # test for predicting 50 words
    if epoch % 1 == 0:
        print(time_since(start_time))
        print('epoch:')
        print(epoch)
        print('loss:')
        print(total_loss[0])
        sentence = ['寄','生']
        predict_word = ''
        count = 0

        while count < 50 :
            word_in = autograd.Variable(torch.cuda.LongTensor([word_to_ix[i] for i in sentence[0:2]]))
            out, outEmbedd = model(word_in)
            _, predict_label = torch.max(out,1)
            predict_word = ix_to_word[predict_label.data[0].item()]
            sentence.insert(0, predict_word)
            count += 1
        print(sentence)
        sentence = []



# saving model state & optimizer state

print('saving model...')
torch.save(model, modelPath_saveTo)
print('state/ epoch:')
print(start_epoch)
state = {
    'epoch': start_epoch + 1,
    'state_dict': model.state_dict(),
    'optimizer' : optimizer.state_dict()
}
torch.save(state, argPath_saveTo)



# predict for 50 words from given words
print('saving done & predicting...')

sentence = ['寄','生']
predict_word = ''
count = 0

while count < 50 :
    word_in = autograd.Variable(torch.cuda.LongTensor([word_to_ix[i] for i in sentence[0:2]]))
    out, outEmbedd = model(word_in)
    _, predict_label = torch.max(out,1)
    predict_word = ix_to_word[predict_label.data[0].item()]
    sentence.insert(0, predict_word)
    count += 1
print(sentence)
sentence = []