# Generative Time Series Model using TimeGAN 
## Context
Time series data is a collection of numerical data points observed or measured in a specific order over a continuous period, such as minutes, days, months, or years. This chronological order allows for the identification of trends, patterns, and seasonality, making it useful for forecasting and making informed decisions. An example of time series data includes stock data such as limit order book (events), where buy and sell orders can fluctuate based on the time during the day. Limit order books (LOB) are split into  ask and bid prices as well as the quantity, with the oldest book at the top while the most recent book sits at the bottom. For this project, the Amazon level 10 orderbook will be used, with ask prices and the required quantity shown from Column 0 to Column 20 (from lowest to highest price) while bid prices and the respective quantity are displayed from Column 21 to column 40 (from highest to lowest price). 
## Limit Book Order (LOB) Problem Space
In the real world, high quality LOB are extremely valuable since they are proprietary, expensive, and limited to large institutions. Due to privacy and confidentiality issues, real orderbook data cannot be shared freely from fear of potentially exposing trading strategies. Regulatory constraint can also restricts data sharing across different jurisdictions and country regulations. Training machine learning models for price prediction and market simulation often require a lot of realistic data, which cannot always be provided. Therefore, synthetic LOB generation provides unlimited, privacy preserving and statistically consistent sequences for model training, testing and simulation. 
## Description of Algorithm
TimeGAN was implemented to generate new realistic time series data for Amazon LOBs. It is trained adversarial and jointly via a learned embedding space with supervised and unsupervised losses.  It combines supervised sequence modelling (RNN autoencoding) with unsupervised adversarial training (GAN) to generate realistic time-series sequences that preserve temporal dynamics. This ensures latent space consistency as the real and synthetic sequences share the same hidden representation space. 
The model uses RNNs (with GRU implementation) to learn temporal dependencies between order flow reactions. The model employs an autoencoder with the embedding module as the encoder stage and recovery module as the decoder stage to learn compressed latent representation of corelated variables. Supervised next step predictions enforces temporal coherence instead of point wise similarity which helps identify trends in ask price and bid prices. This helps capture a more realistic view of the data to create sufficient synthetic LOBs. 
## General Architecture
The model is divided into 5 main modules: \
	1. **Embedder**: converts real sequences into latent representations\
	2. **Recovery**: converts latent representation into reconstructed sequence\
	3. **Generator**: converts random noise into latent representation\
	4. **Supervisor**: helps the generator learn temporal dependencies\
	5. **Discriminator**: distinguishes real and fake latent sequences

The generator and supervisor are trained to fool the discriminator, creating adversarial competition. 

![alt text](image.png)
 
### 1. Embedder
The embedder is the encoder part of the autoencoder. It maps each real sequence into a hidden (latent) space that captures the temporal structure and underlying features of the data
### 2. Recovery
The recovery module is the decoder part of the autoencoder. It reconstructs the original time series from its latent representation. Training the embedder and recovery module together ensures the latent space preserves real temporal information 
### 3. Generator
The generator module learns to produce fake latent representations that look like the real latent space H. Instead of directly generating data, it generates latent sequences which are decoded by the Recovery Module. It is trained adversarial to fool the discriminator module and also supervised by the Supervisor to preserve time dependencies
### 4. Supervisor
The supervisor teaches the generator to produce sequences that follow realistic temporal dynamics.
$$ L_s=|(|H_(t+1)-S(H_t )|)|^2 $$
This acts as a temporal consistency constraint – the generator learns not just to produce realistic points but realistic transitions between timesteps
### 5. Discriminator
The discriminator enforces realism in the latent space. It tries to distinguish between the real latent sequences (from the embedder) and fake latent sequences (from the generator and supervisor)

## File Structure
* **dataset.py**: contains the data loader for loading and preprocessing the data. Uses min max normalisation to scale the data
* **modules.py**: contains the modules required to run the TimeGAN such as Embedder, Recovery, Generator, Supervisor, and Discriminator
* **predict.py**: allows visualisation of 5 representative heatmap visualisation as well as KL divergence graph and visual similarity using SSIM
* **train.py**: contains main function for running dataset.py for data preprocessing, modules.py for training the TimeGAN and predict.py to visualise the results
* **utils.py**: additional helper functions required for training the TimeGAN
## Dependencies Used 
	import torch
	import torch.nn as nn
	import torch.optim as optim
	import numpy as np 
	import matplotlib.pyplot as plt
	from skimage.metric import structural_similarity as ssim
	from scipy.stats import entropy
 
## Hyperparameters
* **hidden_dim** : number of hidden units in GRU layers
* **num_layers** : number of layers in GRU
* **iterations** : number of training iterations (epoch times batches)
* **batch_size** : number of samples per batch
* **z_dim** : dimensionality of latent noise input
* **gamma** : supervised loss weight
* **lr** : learning rate
* **beta1, beta2** : momentum terms for Adam optimiser
* **lambda_stats** : statistics loss weight
* **inst_noise_std** : instance noise added to discriminator inputs to reduce memorisation
* **real_label_smooth** : label smoothing for real labels (improves stability)
