import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import sys
from util import load_data
import os

class word2vec(nn.Module):
    def __init__(self, vocab_size, embedding_size, device):
        super(word2vec, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.input_embed = nn.Embedding(vocab_size, embedding_size, sparse = True)
        self.output_embed = nn.Embedding(vocab_size, embedding_size, sparse = True)
        self.log_sigmoid = nn.LogSigmoid()
        self.device = device
    
    def forward(self, inputs, outputs, neg_samples):
        '''
        parameters: 
            inputs: list [batch]
            outputs: list [batch]
        '''
        
        # in_vec: [batch, feat] out_vec = [batch, feat] 
        in_vec = self.input_embed(inputs)
        out_vec = self.output_embed(outputs)
        pos_loss = self.log_sigmoid(torch.sum( torch.mul(in_vec, out_vec), dim = 1))

        # batch_size = in_vec.shape[0]
        # neg_sample: list [batch, neg_size]
        # neg_sample = torch.LongTensor(batch_size, self.neg_size).random_(0, self.vocab_size).to(self.device)
        neg_vec = self.output_embed(neg_samples)
        neg_product = torch.bmm(neg_vec, in_vec.unsqueeze(2)).squeeze()
        neg_loss = torch.sum( self.log_sigmoid(-neg_product), dim=1)
        return -1 * torch.mean(pos_loss + neg_loss)

    def predict(self, x):
        # x list of idx
        return self.input_embed(torch.tensor(x))

class text_dataset(Dataset):
    def __init__(self, inputs, neg_size, vocab_size):
        '''
        parameters:
            inputs: list of tuple [(x, y)]
        '''
        self.inputs = inputs
        self.neg_size = neg_size
        self.vocab_size = vocab_size

    def __getitem__(self, idx):
        x, y = self.inputs[idx][0], self.inputs[idx][1]
        #n = np.random.randint(self.vocab_size, size=self.neg_size)
        n = torch.LongTensor(self.neg_size).random_(0, self.vocab_size)
        return {'x': x, 'y': y, 'n': n}

    def __len__(self):
        return len(self.inputs)

def save_mat(mat, file_name, idx_to_vocab):
    with open(file_name, 'w') as f:
        for idx, vocab in idx_to_vocab.items():
            line = vocab + ' ' + ' '.join(mat[idx])
            f.write('{}\n'.format(line))

def save_in_out_mat():
    input_embedding_matrix = model.input_embed.weight.data.cpu().numpy().astype(str)
    output_embedding_matrix = model.output_embed.weight.data.cpu().numpy().astype(str)
    save_mat(input_embedding_matrix, 'input_embedding_mat.txt', idx_to_vocab)
    save_mat(output_embedding_matrix, 'output_embedding_mat.txt', idx_to_vocab)

if __name__ == "__main__":
    '''
    args: window size, use_gpu, neg_size
    '''
    train_data, idx_to_vocab = load_data()
    # window_size, neg_size = sys.argv[1], sys.argv[2]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # param
    param = {'optimizer': 'adam', 
             'batch_size': 32768,
             'learning_rate': 1e-2,
             'epoch': 500}

    model = word2vec(vocab_size = max(idx_to_vocab.keys()) + 1, embedding_size = 200, device = device)
    model.to(device)
    print(model)

    if param['optimizer'] == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr = param['learning_rate'])
    elif param['optimizer'] == 'ada':
        opt = torch.optim.Adagrad(model.parameters(), lr = param['learning_rate'])
    elif param['optimizer'] == 'adam':
        opt = torch.optim.SparseAdam(model.parameters(), lr = param['learning_rate'])

    dataset = text_dataset(train_data, neg_size=20, vocab_size=max(idx_to_vocab.keys()))
    train_loader = DataLoader(dataset, batch_size=param['batch_size'], shuffle=False, num_workers=4)
    # tmp = next(iter(train_loader))
    # print(x)
    # loss = model(tmp['x'], tmp['y'])
    # print(loss)

    losses = []
    for e in range(param['epoch']):
        batch_loss = 0
        for d in train_loader:
            x, y, n = d['x'], d['y'], d['n']
            x, y, n = x.to(device).long(), y.to(device).long(), n.to(device).long()

            loss = model(x, y, n)
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_loss += loss.item() 
            sys.stdout.write('batch loss {} \r'.format(loss.item()))
            sys.stdout.flush()

        if e  % 2 == 0 and e != 0:
            save_in_out_mat()
            os.system('python similarity.py > prediction.csv --embedding input_embedding_mat.txt --words ../data/similarity/dev_x.csv')
            os.system('python evaluate.py --predict prediction.csv --development ../data/similarity/dev_y.csv')
        losses.append(batch_loss / len(x))
        print('Epoch: {}'.format(e))
        print('loss: {}'.format(losses[-1]))



    # generate embedding matrix
    save_in_out_mat()






