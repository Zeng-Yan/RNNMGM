import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from configs import DEVICE


class RNNPredictor(nn.Module):

    def __init__(self, **args):
        super(RNNPredictor, self).__init__()

        size_of_oh = args['size_of_oh']
        units_of_rnn = args['units_of_rnn']
        layers_of_rnn = args['layers_of_rnn']
        units_of_nn = args['units_of_nn']
        dropout_rate = args['dropout_rate']

        self.param = args
        self.lstm = nn.LSTM(input_size=size_of_oh, hidden_size=units_of_rnn, num_layers=layers_of_rnn,
                            dropout=dropout_rate, batch_first=True)
        self.output1 = nn.Linear(units_of_rnn, units_of_nn)
        self.output2 = nn.Linear(units_of_nn, 1)
        if args['activation'] == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

    def init_states(self, batch_size):
        layers_of_rnn = self.param['layers_of_rnn']
        units_of_rnn = self.param['units_of_rnn']
        hidden = [torch.zeros(layers_of_rnn, batch_size, units_of_rnn).to(DEVICE),
                  torch.zeros(layers_of_rnn, batch_size, units_of_rnn).to(DEVICE)]
        return hidden

    def forward(self, inp, hidden, is_train=True):
        output, hidden = self.lstm(inp, hidden)
        (h, cell) = hidden
        output = h[-1, :, :]
        output = self.activation(self.output1(output))
        output = F.dropout(output, p=self.param['dropout_rate'])
        output = self.output2(output)
        return output, hidden


def train(model, optimizer, train_loader, valid_x, valid_y, epochs, fq_of_save, patience=5, name='reg'):
    print('Start training')
    print('Epoch, Training loss, Valid loss')
    record_path = os.getcwd() + '/record/'
    if not os.path.exists(record_path):
        os.makedirs(record_path)

    records = []
    patience_count = 0
    valid_loss_pre = 999
    for epoch in range(1, epochs + 1):
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            model.train()
            (x_padded, lengths, y) = data
            x_packed = rnn_utils.pack_padded_sequence(x_padded, lengths, batch_first=True)
            x_packed = x_packed.to(DEVICE)
            y = y.float().to(DEVICE)
            hidden = model.init_states(x_padded.size(0))
            optimizer.zero_grad()
            y_, _ = model(x_packed, hidden)
            loss = F.mse_loss(y_.squeeze(1), y, reduction='sum')
            loss.backward()
            optimizer.step()
            train_loss += loss

        train_loss = train_loss / len(train_loader.dataset)
        y_predict = predict(model, valid_x)
        valid_y = torch.tensor(valid_y, dtype=torch.float).to(DEVICE)
        valid_loss = F.mse_loss(y_predict, valid_y, reduction='mean')

        if epoch < 2:
            valid_loss_pre = valid_loss
        else:
            if (valid_loss >= valid_loss_pre) and (valid_loss < 0.1):
                patience_count += 1
            else:
                patience_count = 0
            valid_loss_pre = valid_loss
        if patience_count >= patience:
            print(f"Early Stop in epoch {epoch} with patience {patience}")
            break

        record = [epoch, train_loss.item(), valid_loss.item()]
        records.append(record)
        print(record, ',')

        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epochs': epoch}
        torch.save(state, 'record/reg.pth')

        if fq_of_save == 0:
            continue
        if epoch % fq_of_save == 0:
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epochs': epoch}
            torch.save(state, 'record/{}-{:04d}-{:.4f}.pth'.format(name, epoch, train_loss))

    return records


def predict(model, x):
    model.eval()
    y_predict = torch.zeros(len(x)).to(DEVICE)
    for index in range(len(x)):
        x_ = x[index].unsqueeze(0).to(DEVICE)
        y_, _ = model(x_, model.init_states(1), False)
        y_predict[index] = y_.to(DEVICE)
    return y_predict
