import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from configs import DEVICE


def criterion_by_masked(recon: torch.Tensor, target: torch.Tensor, mask_tensor: torch.Tensor):

    n = mask_tensor.size(dim=0)
    index = [idx for idx in range(n) if mask_tensor[idx] == 1]
    index = torch.tensor(index, dtype=torch.long, device=DEVICE)

    target = target.index_select(dim=0, index=index)
    recon = recon.index_select(dim=0, index=index)

    loss = F.cross_entropy(recon, target.long(), reduction='sum')

    return loss


class NNLM(nn.Module):

    def __init__(self, **args):
        super(NNLM, self).__init__()
        size_of_oh = args['size_of_oh']
        units_of_rnn = args['units_of_rnn']
        layers_of_rnn = args['layers_of_rnn']
        dropout_rate = args['dropout_rate']
        self.param = args
        self.lstm = nn.LSTM(input_size=size_of_oh, hidden_size=units_of_rnn, num_layers=layers_of_rnn,
                            dropout=dropout_rate, batch_first=True)
        self.linear = nn.Linear(units_of_rnn, size_of_oh)

    def init_states(self, batch_size):
        layers_of_rnn = self.param['layers_of_rnn']
        units_of_rnn = self.param['units_of_rnn']
        hidden = [torch.zeros(layers_of_rnn, batch_size, units_of_rnn).to(DEVICE),
                  torch.zeros(layers_of_rnn, batch_size, units_of_rnn).to(DEVICE)]
        return hidden

    def forward(self, inp, hidden):
        output, hidden = self.lstm(inp, hidden)
        output = output.reshape(-1, self.param['units_of_rnn'])
        output = self.linear(output)
        return output, hidden


def train(model, optimizer, data_loader, epochs, fq_of_save, name):
    print('epoch, train loss')

    record_path = os.getcwd() + '/record/'
    if not os.path.exists(record_path):
        os.makedirs(record_path)

    records = []
    for epoch in range(1, epochs + 1):
        train_loss = 0
        for batch_idx, data in enumerate(data_loader):
            model.train()
            oh_pre, lb, l_of_len = data
            oh_pre = oh_pre.to(DEVICE)
            lb = lb.long().to(DEVICE)
            l_of_len = l_of_len.byte().to(DEVICE)
            lb = lb.reshape(-1)
            l_of_len = l_of_len.reshape(-1)
            hidden = model.init_states(oh_pre.size(0))
            optimizer.zero_grad()
            recon, _ = model(oh_pre, hidden)
            loss = criterion_by_masked(recon, lb, l_of_len)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss

        train_loss = train_loss / len(data_loader.dataset)
        record = [epoch, train_loss.item()]
        records.append(record)
        print(record, ',')

        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epochs': epoch}
        torch.save(state, '{}.pth'.format(name))

        if epoch % fq_of_save == 0:
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epochs': epoch}
            torch.save(state, 'record/{}-{:04d}-{:.4f}.pth'.format(name, epoch, train_loss))

    return records


def sample(model, token_list, number_of_sample):

    model.eval()

    def token2tensor(t):
        tensor = np.zeros(len(token_list))
        tensor[token_list.index(t)] = 1
        tensor = torch.from_numpy(tensor).float().to(DEVICE)
        return tensor

    smi_list = []

    for i in range(number_of_sample):
        predict_token = token2tensor('^')
        end_token = token2tensor(' ')
        smi = ''
        hidden = model.init_states(1)
        while (not predict_token.equal(end_token)) and len(smi) < 120:
            output, hidden = model(predict_token.reshape(1, 1, -1), hidden)
            output = F.softmax(output)
            output = output.data.reshape(-1)
            idx = torch.multinomial(output, 1)
            token = token_list[idx]
            smi += token
            predict_token = token2tensor(token)
        smi_list.append(smi)

    return smi_list

