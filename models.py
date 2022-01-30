import collections
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys,random,math
from collections import Counter
import numpy as np
from bpe import *
import torch.optim as optim

#Transformer
class Transformer(nn.Module):
    def __init__(self, heads, src_len, trg_len, d_model, vocab_size, n_blocks = 6):
        super(Transformer, self).__init__()

        self.heads = heads
        self.src_len = src_len
        self.trg_len = trg_len
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.vocab_size = vocab_size 

        self.encoders = nn.ModuleList([Encoder(self.heads, self.src_len, self.d_model) for i in range(self.n_blocks)]) 
        self.decoders = nn.ModuleList([Decoder(self.heads, self.trg_len, self.d_model) for i in range(self.n_blocks)]) 

        self.pe = PositionalEncoding(self.src_len, self.d_model)
        self.embed = nn.Embedding(self.vocab_size, self.d_model)

        self.linear = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, src_idx, trg_idx):
        bs = src_idx.size(0)
        src_out = self.pe()[:self.src_len,:] + self.embed(src_idx)

        for i in range(len(self.encoders)):
            src_out = self.encoders[i](src_out)

        trg_out = self.pe()[:self.trg_len,:] + self.embed(trg_idx)
        for i in range(len(self.decoders)):
            trg_out = self.decoders[i](trg_out, src_out)

        trg_out = self.linear(trg_out.view(bs * self.trg_len, -1))
        trg_out = F.softmax(trg_out, dim = -1)
        trg_out = trg_out.view(bs, self.trg_len, self.vocab_size)
        return trg_out

#class GPT
class GPT(nn.Module):
    def __init__(self , heads, trg_len, d_model, vocab_size, n_blocks = 4):
        super(GPT, self).__init__()

        self.heads = heads
        self.trg_len = trg_len
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.vocab_size = vocab_size 

        #self.encoders = nn.ModuleList([Encoder(self.heads, self.src_len, self.d_model) for i in range(self.n_blocks)]) 
        self.decoders = nn.ModuleList([Decoder(self.heads, self.trg_len, self.d_model) for i in range(self.n_blocks)]) 

        self.pe = PositionalEncoding(self.trg_len + 100, self.d_model)
        self.embed = nn.Embedding(self.vocab_size, self.d_model)

        self.linear = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, idx):
        bs = idx.size(0)

        trg_out = self.pe()[:self.trg_len,:] + self.embed(idx)
        for i in range(len(self.decoders)):
            trg_out = self.decoders[i](trg_out, trg_out)

        trg_out = self.linear(trg_out.view(bs * self.trg_len, -1))
        trg_out = F.softmax(trg_out, dim = -1)
        trg_out = trg_out.view(bs, self.trg_len, self.vocab_size)
        return trg_out

#Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        self.register_buffer('PE', self.generate_pe())

    def forward(self):
        return self.PE

    def generate_pe(self):
        pos_enc = torch.zeros((self.seq_len, self.d_model))
        den = torch.pow(10000, 2 * (torch.arange(pos_enc.size(1)//2)) / self.d_model)
        pos_enc[:, 0::2] = torch.sin(torch.arange(pos_enc.size(0)).unsqueeze(1)/den)
        pos_enc[:, 1::2] = torch.cos(torch.arange(pos_enc.size(0)).unsqueeze(1)/den)
        return pos_enc
#Encoder
class Encoder(nn.Module):
    def __init__(self, heads,seq_len, d_model):
        super(Encoder, self).__init__()
        self.heads = heads
        self.d_model = d_model
        self.multihead_attn = MultiHeadAttention(self.heads, self.d_model)
        self.ln1 = nn.BatchNorm1d(seq_len)
        self.ffnn = FFNN(self.d_model)
        self.ln2 = nn.BatchNorm1d(seq_len)
        self.dropout1 = nn.Dropout(0.6)
        self.dropout2 = nn.Dropout(0.8)
    def forward(self, inputs):
        inputs = self.ln1(self.dropout1(self.multihead_attn(inputs, inputs, inputs)) + inputs)
        inputs = self.ln2(self.dropout2(self.ffnn(inputs)) + inputs)
        return inputs

#Decoder
class Decoder(nn.Module):
    def __init__(self, heads, trg_len, d_model):
        super(Decoder, self).__init__()

        self.heads = heads
        self.d_model = d_model
        self.trg_len = trg_len 

        self.mask_attn = MultiHeadAttention(self.heads, self.d_model)
        self.ln1 = nn.BatchNorm1d(trg_len)

        self.multihead_attn = MultiHeadAttention(self.heads, self.d_model)
        self.ln2 = nn.BatchNorm1d(trg_len)

        self.ffnn = FFNN(self.d_model)
        self.ln3 = nn.BatchNorm1d(trg_len)

        self.dropout1 = nn.Dropout(0.8)
        self.dropout2 = nn.Dropout(0.8)
        self.dropout3 = nn.Dropout(0.8)
        self.generate_mask()

    def forward(self, inputs, keys):

        inputs = self.ln1(self.dropout1(self.multihead_attn(inputs, inputs, inputs, mask = self.mask)) + inputs)
        inputs = self.ln2(self.dropout2(self.multihead_attn(inputs, keys, keys)) + inputs)
        inputs = self.ln3(self.dropout3(self.ffnn(inputs)) + inputs)
        return inputs

    def generate_mask(self):
        mask = torch.ones((self.trg_len, self.trg_len))
        mask = torch.triu(mask, diagonal = 1) * -100000
        self.register_buffer('mask', mask)
#FFNN
class FFNN(nn.Module):
    def __init__(self, d_model):
        super(FFNN, self).__init__()
        self.d_model = d_model
        self.ln1 = nn.Linear(self.d_model, self.d_model * 4)
        self.ln2 = nn.Linear(self.d_model * 4, self.d_model)

    def forward(self, x):
        inputs = F.relu(self.ln1(x))
        inputs = self.ln2(inputs) + x
        return inputs

#Multihead Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):
        super(MultiHeadAttention, self).__init__()

        self.heads = heads
        self.d_model = d_model

        assert self.d_model % self.heads == 0, 'heads has to be a multiple of d_model'
        self.query = nn.Linear(self.d_model, self.d_model)
        self.key = nn.Linear(self.d_model, self.d_model)
        self.value = nn.Linear(self.d_model, self.d_model)

        self.ln = nn.Linear(self.d_model, self.d_model)

    def forward(self, query, key, value, mask = None):
        bs = query.size(0)

        #inputs = inputs.view(-1, self.d_model)
        query = self.query(query.view(-1, self.d_model)).view(bs, -1, self.heads, self.d_model//self.heads)
        key = self.key(key.view(-1, self.d_model)).view(bs, -1, self.heads, self.d_model//self.heads)
        value = self.value(value.view(-1, self.d_model)).view(bs, -1, self.heads, self.d_model//self.heads)

        
        #print(query.shape)
        query = query.permute(0,2,1,3).contiguous().view(bs * self.heads, -1, self.d_model//self.heads)
        key = key.permute(0,2,1,3).contiguous().view(bs * self.heads, -1, self.d_model//self.heads)
        value = value.permute(0,2,1,3).contiguous().view(bs * self.heads, -1, self.d_model//self.heads)

        inputs = sccaled_dot_product_attention(query, key, value, mask) 
        inputs = inputs.view(bs, self.heads, -1, self.d_model//self.heads).permute(0,2,1,3).contiguous()

        inputs = self.ln(inputs.view(bs * inputs.size(1), self.d_model)).view(bs, -1, self.d_model)
        return inputs

#scaled dot product attention
def sccaled_dot_product_attention(query, key, value, mask):
    if mask is not None:
        attn = torch.bmm(query, key.transpose(-1,-2))/torch.sqrt(torch.FloatTensor([key.size(-1)]).to(key.device)) + mask.to(key.device)
    else:
        attn = torch.bmm(query, key.transpose(-1,-2))/torch.sqrt(torch.FloatTensor([key.size(-1)]).to(key.device))

    attn = torch.softmax(attn, dim = -1)
    
    output = torch.bmm(attn, value)
    return output


def words2indices(sentence):
    idx = list()
    for word in sentence:
        idx.append(word2index[word])
    return idx


def generate_data():
    f = open('data/shakespeare.txt','r')
    raw = f.readlines()
    f.close()

    tokens = list()
    tk = []
    for line in raw:
        info = line.lower().replace("\n","").split(" ")[1:]
        if (len(tk) + len(info) <= 510):
            tk = tk + info
            #print(info)
            #print(tk)
        else:
            tokens.append(tk)
            tk = [] 
            tk = tk + info
    new_tokens = list()

    for line in tokens:
        new_tokens.append(['<Strt>'] +['-'] * (510 - len(line)) + line + ['<END>'])
    
    trg_tokens = list()
    for line in tokens:
        trg_tokens.append(['-'] * (510 - len(line)) + line + ['<END>'] + ['<END>'])
    
    tokens = new_tokens
    vocab = set()
    for sent in tokens:
        for word in sent:
            vocab.add(word)
    vocab = list(vocab)
    word2index = {}
    for i,word in enumerate(vocab):
        word2index[word]=i


    indices = list()
    for line in tokens:
        idx = list()
        for w in line:
            idx.append(word2index[w])
        indices.append(idx)

    trg_idx = list()
    for line in trg_tokens:
        idx = list()
        for w in line:
            idx.append(word2index[w])
        trg_idx.append(idx)

    data = np.array(indices)
    trg_data = np.array(trg_idx)
    return data, trg_data

def train_loader(data, trg_data, bs):
    
    data = torch.from_numpy(data)
    trg_data = torch.from_numpy(trg_data)
    len_ = data.size(0)
    for i in range(data.size(0)):
        idx = torch.randint(0, int(len_), (bs, 1))
        yield data[idx].squeeze(1), trg_data[idx].squeeze(1)


if __name__ == "__main__":
    #pe = PositionalEncoding(512, 1024)
    #plt.imshow(pe().detach())
    #plt.show()
    
    #print(len(vocab))
    data, trg_data = generate_data()
    model2 = GPT(8, 512, 1024, int(torch.max(torch.from_numpy(data))))
    optimizer = torch.optim.Adam(model2.parameters(), lr = 0.01)
    loss_fn = nn.BCELoss()
    
    for idx, trg_idx in train_loader(data, trg_data, 4):
        
        out = model2(idx)
        
        one_hot = torch.zeros(out.shape)
        target = one_hot.scatter(-1, trg_idx.unsqueeze(-1).long(), 1.0)
        optimizer.zero_grad()

        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()

        print(loss.item())

    plt.imshow(out[1,:,:].detach())
    plt.show()
    print(out.shape)

    plt.show()