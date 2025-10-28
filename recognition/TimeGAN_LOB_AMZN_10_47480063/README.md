# Generative Time Series Model using TimeGAN that generates synthetic sequences of limit order books for amazon level 10 data

## Description of Algorithm
1 paragraph

## Problem it solves
1 paragraph

## How it works
1 paragraph
add figure/visualisation


## Model Architecture
dataset.py
modules.py
predict.py
train.py
utils.py

## Dependencies required 
1 paragraph
It should also list any dependencies required, including versions and address reproduciblility of results,
 if applicable.

## Example inputs, outputs and plots of your algorithm
 5. Describe any specific pre-processing you have used with references if any. Justify your training, validation
 and testing splits of the data.

## Hyperparameters

## GPU
NVIDIA A100
A100 High RAM used

## Epoch used

## Total training Time

## Training Process
Target: KL <= 0.1 for good similarity  
Target: SSIM > 0.6 for good visual similarity

Iteration 1: 
KL Divergence (Spread): 1.8591 
KL Divergence (Midprice Return): 14.0299 

SSIM (Heatmap Visual Similarity): 0.9988 
High SSIM suggests overfitting to the data

Training Time: 36 min 04 seconds

### Changes made
Added dropout in the generative path of the model. added dropout = 0.3 in generator and supervisor but not in discriminator or recovery
increased training epoch from 1000 to 5000

Iteration 3: 
KL Divergence (Spread): 1.1912
KL Divergence (Midprice Return): 6.4541
Target: KL ≤ 0.1 for good similarity

SSIM (Heatmap Visual Similarity): 0.9998
Target: SSIM > 0.6 for good visual similarity



