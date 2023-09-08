# Fence_GAN_torch
PyTorch implementation of Fence GAN [(link to paper)](https://arxiv.org/abs/1904.01209) <br>
Official implementation (in TensorFlow) [(link)](https://github.com/phuccuongngo99/Fence_GAN/tree/master)

## Prerequisites
1. Windows OS
2. Python 3
3. PyTorch

## Installation
1. Clone repository
    ```
    git clone https://github.com/amothytintoni/Fence_GAN_torch.git
    ```

2. Install necessary libraries
    ```
    pip3 install -r requirements.txt

    or

    conda create --name fence_gan_env --file requirements.txt
    ```

## Anomaly Detection
Check results and plots under `result` folder
### 2D Synthetic Dataset
    
    python3 2D_fgan.py
    
### MNIST
    python3 main.py --dataset mnist --ano_class 0 --epochs 100 --alpha 0.1 --beta 30 
    --gamma 0.1 --batch_size 200 --pretrain 15 --d_lr 1e-5 --g_lr 2e-5 --v_freq 4 
    --latent_dim 200 --evaluation 'auprc'

### CIFAR10
    python3 main.py --dataset cifar10 --ano_class 0 --epochs 150 --alpha 0.5 --beta 10 
    --gamma 0.5 --batch_size 128 --pretrain 15 --d_lr 1e-4 --g_lr 1e-3 --v_freq 1 
    --latent_dim 256 --evaluation 'auroc'

### KDD99
Unzip the KDD99_Final.zip and then run Fence_GAN.py. Hyperparameters are set as global variables in the Fence_GAN.py file

### More training option
Enter `python3 main.py -h` for more training options
```
    usage: Train your Fence GAN [-h] [--dataset DATASET] [--ano_class ANO_CLASS]
                                [--epochs EPOCHS] [--beta BETA] [--gamma GAMMA]
                                [--alpha ALPHA] [--batch_size BATCH_SIZE]
                                [--pretrain PRETRAIN] [--d_l2 D_L2] [--d_lr D_LR]
                                [--g_lr G_LR] [--v_freq V_FREQ] [--seed SEED]
                                [--evaluation EVALUATION]
                                [--latent_dim LATENT_DIM]

    optional arguments:
      -h, --help            show this help message and exit
      --dataset         mnist | cifar10
      --ano_class       1 anomaly class
      --epochs          number of epochs to train
      --beta            beta
      --gamma           gamma
      --alpha           alpha
      --batch_size 
      --pretrain        number of pretrain epoch
      --d_l2            L2 Regularizer for Discriminator
      --d_lr            learning_rate of discriminator
      --g_lr            learning rate of generator
      --v_freq          epoch frequency to evaluate performance
      --seed            numpy and tensorflow seed
      --evaluation      'auprc' or 'auroc'
      --latent_dim      Latent dimension of Gaussian noise input to Generator
  ```
  
## Citation
  ```
  @article{ngo2019,
      author    = {Cuong Phuc Ngo and Amadeus Aristo Winarto and Connie Khor Li Kou and
                   Sojeong Park and Farhan Akram and Hwee Kuan Lee},
      title     = {Fence GAN: Towards Better Anomaly Detection},
      year      = {2019},
      url       = {https://arxiv.org/pdf/1904.01209.pdf},
      archivePrefix = {arXiv}
  }
  ```
