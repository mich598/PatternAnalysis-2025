"""
containing the data loader for loading and preprocessing your data

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Github Link: https://github.com/jsyoon0823/TimeGAN/blob/master/data_loading.py

data_loading.py

(0) MinMaxScaler: Min Max normalizer
(1) real_data_loading: Load and preprocess real data
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
"""

## Necessary Packages
import numpy as np


def MinMaxScaler(data):
  """Min Max normalizer.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
  """
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  norm_data = numerator / (denominator + 1e-7)
  return norm_data
    

def real_data_loading (data_name, seq_len):
    """Load and preprocess real-world datasets.

    Args:
    - data_name: message or orderbook
    - seq_len: sequence length

    Returns:
    - data: preprocessed data.
    """  
    assert data_name in ['stock','orderbook']

    # Note: data is in chronological order (oldest entry at the top, newest at the bottom)
    if data_name == 'stock':
        ori_data = np.loadtxt('amzn_message_data.csv', delimiter = ",")
    elif data_name == 'orderbook':
        ori_data = np.loadtxt('amzn_orderbook_data.csv', delimiter = ",")

    # Normalize the data
    ori_data = MinMaxScaler(ori_data)

    # Preprocess the dataset
    temp_data = []    
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)
        
    # Mix the datasets (to make it similar to independent and identically distributed)
    idx = np.random.permutation(len(temp_data))    
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
        
    return data