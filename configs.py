import torch
import platform

if 'Windows' not in platform.platform():
    import matplotlib
    matplotlib.use('Agg')
    print('[WARNING] matplotlib use Agg.')

# settings
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BOS = '^'
EOS = ' '
SINGLE_TOKENIZE = False
COLORS = {
    'black': '#595959',
    'red': '#D1603D',
    'yellow': '#FFAD05',
    'green': '#A8C256',
    'hunter': '#33673B',
    'tur': '#25CED1',
    'blue': '#1982C4',
    'white': '#F1E0C5',
}

# hyper params of CLM
CLM_N_RNN_LAYERS = 3
CLM_N_RNN_UNITS = 256
CLM_DROPOUT_RATE = 0.2
CLM_BATCH_SIZE = 128
CLM_LR_RATE = 0.001
MAX_SEQ_LEN = 200

# hyper params of REG
REG_N_RNN_LAYERS = 2
REG_N_RNN_UNITS = 256
REG_N_NN_UNITS = 256
REG_NN_ACTIVATION = 'tanh'
REG_DROPOUT_RATE = 0.2
REG_BATCH_SIZE = 128
REG_LR_RATE = 0.001




