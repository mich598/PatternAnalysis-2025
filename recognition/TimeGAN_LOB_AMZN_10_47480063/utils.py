"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Github Link: https://github.com/jsyoon0823/TimeGAN/blob/master/data_loading.py

-----------------------------

Based on utils.py

(1) train_test_divide: Divide train and test data for both original and synthetic data.
(2) extract_time: Returns Maximum sequence length and each sequence length.
(3) rnn_cell: Basic RNN Cell.
(4) random_generator: random vector generator
(5) batch_generator: mini-batch generator
"""

## Necessary Packages
import numpy as np
import torch
import torch.nn as nn


def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
    """Divide train and test data for both original and synthetic data.
    
    Args:
        - data_x: original data
        - data_x_hat: generated data
        - data_t: original time
        - data_t_hat: generated time
        - train_rate: ratio of training data from the original data
    """
    # Divide train/test index (original data)
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]
    
    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]      
    
    # Divide train/test index (synthetic data)
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]
    
    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]
    
    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time(data):
    """Returns Maximum sequence length and each sequence length.
    
    Args:
        - data: original data
        
    Returns:
        - time: extracted time information
        - max_seq_len: maximum sequence length
    """
    time = []
    max_seq_len = 0
    for i in range(len(data)):
        seq_len = len(data[i][:, 0])
        max_seq_len = max(max_seq_len, seq_len)
        time.append(seq_len)
    return time, max_seq_len


def rnn_cell(module_name, hidden_dim):
    """Create a PyTorch RNN cell.
    
    Args:
        - module_name: 'gru', 'lstm', or 'lstmLN'
        - hidden_dim: number of hidden units
        
    Returns:
        - rnn_cell: RNN Cell module
    """
    assert module_name in ['gru', 'lstm', 'lstmLN']
    
    if module_name == 'gru':
        rnn_cell = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
    elif module_name == 'lstm':
        rnn_cell = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)
    elif module_name == 'lstmLN':
        # LayerNorm LSTM: implement manually using wrapper
        class LayerNormLSTMCell(nn.Module):
            def __init__(self, input_size, hidden_size):
                super().__init__()
                self.lstm = nn.LSTMCell(input_size, hidden_size)
                self.ln_h = nn.LayerNorm(hidden_size)
                self.ln_c = nn.LayerNorm(hidden_size)

            def forward(self, x, states):
                h, c = self.lstm(x, states)
                h = self.ln_h(h)
                c = self.ln_c(c)
                return h, c

        rnn_cell = LayerNormLSTMCell(hidden_dim, hidden_dim)

    return rnn_cell


def random_generator(batch_size, z_dim, T_mb, max_seq_len):
    """Random vector generation.
    
    Args:
        - batch_size: size of the random vector
        - z_dim: dimension of random vector
        - T_mb: time information for the random vector
        - max_seq_len: maximum sequence length
        
    Returns:
        - Z_mb: generated random vector
    """
    Z_mb = []
    for i in range(batch_size):
        temp = np.zeros([max_seq_len, z_dim])
        temp_Z = np.random.uniform(0., 1., [T_mb[i], z_dim])
        temp[:T_mb[i], :] = temp_Z
        Z_mb.append(temp_Z)
    return Z_mb


def batch_generator(data, time, batch_size):
    """Mini-batch generator.
    
    Args:
        - data: time-series data
        - time: time information
        - batch_size: number of samples per batch
        
    Returns:
        - X_mb: time-series data batch
        - T_mb: time information batch
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]
    
    X_mb = [data[i] for i in train_idx]
    T_mb = [time[i] for i in train_idx]
    
    return X_mb, T_mb