import torch
import platform

if 'Windows' not in platform.platform():
    import matplotlib
    matplotlib.use('Agg')
    print('[WARNING] matplotlib use Agg.')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BOS = '^'
EOS = ' '
SINGLE_TOKENIZE = False


CLM_N_RNN_LAYERS = 3
CLM_N_RNN_UNITS = 256
CLM_DROPOUT_RATE = 0.2
CLM_BATCH_SIZE = 128
CLM_LR_RATE = 0.001
MAX_SEQ_LEN = 200


REG_N_RNN_LAYERS = 2
REG_N_RNN_UNITS = 256
REG_N_NN_UNITS = 256
REG_NN_ACTIVATION = 'tanh'
REG_DROPOUT_RATE = 0.2
REG_BATCH_SIZE = 128
REG_LR_RATE = 0.001


colors = {
    'blue': '#c4ccd7',
    'b_dark': '#8797a6',
    'green': '#b7c6b3',
    'g_dark': '#7d8b72',
    'red': '#edced3',
    'r_dark': '#945657',
    'orange': '#FDBE6F',
    'o_dark': '#c8b9a6',
    'purple': '#f1e6f7',
    'p_dark': '#ccc1d2',
    'yellow': '#fae9d5',
    'black': '#656565'
}





