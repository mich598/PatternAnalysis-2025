"""
containing the source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
sure to plot the losses and metrics during training

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Github Link: https://github.com/jsyoon0823/TimeGAN/blob/master/data_loading.py

Based on main_timegan.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 1. TimeGAN model
from modules import timegan
# 2. Data loading
from dataset import real_data_loading
# 3. Metrics
from predict import visualization

def main ():
    """Main function for timeGAN experiments.

    Args:
        - data_name: sine, stock, or energy
        - seq_len: sequence length
        - Network parameters (should be optimized for different datasets)
        - module: gru, lstm, or lstmLN
        - hidden_dim: hidden dimensions
        - num_layer: number of layers
        - iteration: number of training iterations
        - batch_size: the number of samples in each batch
        - metric_iteration: number of iterations for metric computation

    Returns:
        - ori_data: original data
        - generated_data: generated synthetic data
        - metric_results: discriminative and predictive scores
    """
    ## Data loading
    ori_data = real_data_loading('orderbook', 24)

    print('orderbook dataset is ready.')

    ## Synthetic data generation by TimeGAN
    # Set newtork parameters
    parameters = dict()
    parameters['module'] = 'gru'
    parameters['hidden_dim'] = 24
    parameters['num_layer'] = 3
    parameters['iterations'] = 100
    parameters['batch_size'] = 128

    generated_data = timegan(ori_data, parameters)
    print('Finish Synthetic Data Generation')

    ## Performance metrics
    # Output initialization
    metric_results = dict()

    # 3. Visualization with KL divergence and SSIM
    visualization(ori_data, generated_data)
    visualization(ori_data, generated_data)

    ## Print discriminative and predictive scores
    print(metric_results)

    return ori_data, generated_data, metric_results


if __name__ == '__main__':
    # Calls main function
    ori_data, generated_data, metrics = main()
