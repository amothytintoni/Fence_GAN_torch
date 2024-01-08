# torch version

import torch
from torch import nn
import os
import json
import random
from tqdm import trange
from collections import OrderedDict
import time
from copy import deepcopy

import numpy as np
from numpy.random import seed
# from tensorflow import set_random_seed
# import keras.backend as K

from model import *
from data import load_data
from visualize import show_images, compute_au, histogram
from custom_losses import *


def set_trainability(model, trainable=False):
    model.requires_grad_ = trainable
    for param in model.parameters():
        param.requires_grad = trainable


def noise_data(n_samples, latent_dim):
    return np.random.normal(0, 1, [n_samples, latent_dim])


def D_data(n_samples, G, mode, x_train, latent_dim):
    # Feeding training data for normal case
    if mode == 'normal':
        sample_list = random.sample(list(range(np.shape(x_train)[0])), n_samples)
        x_normal = torch.tensor(x_train[sample_list, ...]).float()
        y1 = torch.tensor(np.ones(n_samples)).float()

        return x_normal, y1

    # Feeding training data for generated case
    if mode == 'gen':
        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        noise = noise_data(n_samples, latent_dim)
        noise = torch.tensor(noise).float().to(device)

        G = G.to(device)

        x_gen = G(noise)
        y0 = torch.tensor(np.zeros(n_samples)).float().to(device)

        return x_gen, y0


def pretrain(args, G, D, GAN, x_train, x_test, y_test, x_val, y_val, dopt):
    # Pretrain discriminator
    ###Generator is not trained
    print("===== Start of Pretraining =====")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'\ndevice: {device}')
    batch_size = args.batch_size
    pretrain_epoch = args.pretrain
    latent_dim = args.latent_dim
    for e in range(pretrain_epoch):
        with trange(x_train.shape[0]//batch_size, ascii=True, desc='Pretrain_Epoch {}'.format(e+1)) as t:
            for step in t:

                D.train()
                dopt.zero_grad()
                set_trainability(D, True)
                # K.set_value(gamma, [1])
                x, y = D_data(batch_size, G, 'normal', x_train, latent_dim)
                D = D.to(device)
                x = x.float().to(device)
                y = y.float().to(device)
                output = D(x)
                loss1 = args.gamma * D_loss(y, output)

                set_trainability(D, True)
                # K.set_value(gamma, [args.gamma])
                x, y = D_data(batch_size, G, 'gen', x_train, latent_dim)
                D = D.to(device)
                x = x.float().to(device)
                y = y.float().to(device)
                output = D(x)
                loss2 = args.gamma * D_loss(y, output)

                loss = (loss1+loss2)/2
                loss.backward()
                dopt.step()

                if args.progress_bar:
                    t.set_postfix(D_loss=loss)
        print("\tDisc. Loss: {:.4f}".format(loss))
    print("===== End of Pretraining =====")


def train(args, G, D, GAN, x_train, x_test, y_test, x_val, y_val, dopt, gopt):
    # Adversarial Training
    now = time.time()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    epochs = args.epochs
    batch_size = args.batch_size
    v_freq = args.v_freq
    ano_class = args.ano_class
    latent_dim = args.latent_dim
    evaluation = args.evaluation

    if not os.path.exists('./result/{}/'.format(args.dataset)):
        os.makedirs('./result/{}/'.format(args.dataset))
    result_path = './result/{}'.format(args.dataset)  # ,
    # len(os.listdir('./result/{}/'.format(args.dataset))))
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    d_loss = []
    g_loss = []
    best_val = 0
    best_test = 0

    print('===== Start of Adversarial Training =====')
    for epoch in range(epochs):
        # try:
        with trange(x_train.shape[0]//batch_size, ascii=True, desc='Epoch {}'.format(epoch+1)) as t:
            for step in t:

                D.train()
                dopt.zero_grad()

                # Train Discriminator
                loss_temp = []

                set_trainability(D, True)
                # K.set_value(gamma, [1])
                x, y = D_data(batch_size, G, 'normal', x_train, latent_dim)

                D = D.to(device)
                x = x.float().to(device)
                y = y.float().to(device)

                realdata = deepcopy(x)
                output = D(x)
                loss1 = args.gamma * D_loss(y, output)
                loss_temp.append(loss1)

                set_trainability(D, True)
                # K.set_value(gamma, [args.gamma])
                x, y = D_data(batch_size, G, 'gen', x_train, latent_dim)
                D = D.to(device)
                x = x.float().to(device)
                y = y.float().to(device)
                output = D(x)
                loss2 = args.gamma * D_loss(y, output)
                loss_temp.append(loss2)

                loss_combined = (loss1+loss2)/2
                loss_combined.backward()
                dopt.step()

                d_loss.append(sum(loss_temp)/len(loss_temp))

                # Train Generator
                GAN.train()
                G.train()
                gopt.zero_grad()

                set_trainability(D, False)
                x = noise_data(batch_size, latent_dim)
                y = np.zeros(batch_size)
                y[:] = args.alpha
                GAN = GAN.to(device)
                x = torch.tensor(x).float().to(device)
                y = torch.tensor(y).float().to(device)

                output = GAN(D, G, x)
                G_out = GAN.get_G_out()
                gan_criterion = com_conv(G_out, args.beta, 2, args.bm)
                gan_loss = gan_criterion(y, output)

                ol_loss = 0
                if args.ol:
                    ol_loss = outline_loss(G_out, realdata, args.omega, 2, args.verif)
                    if (epoch + 1) % v_freq == 0:
                        print('GAN loss=', gan_loss, 'Outline Loss=', ol_loss)
                    gan_loss += args.kappa * ol_loss

                g_loss.append(gan_loss)

                gan_loss.backward()
                gopt.step()

                if args.progress_bar:
                    t.set_postfix(G_loss=g_loss[-1], D_loss=d_loss[-1],
                                  O_loss=args.kappa * ol_loss)
        # except KeyboardInterrupt:  # hit control-C to exit and save video there
        #     break

        D.eval()
        GAN.eval()
        G.eval()

        if (epoch + 1) % v_freq == 0:
            val, test = compute_au(D, G, GAN, x_val, y_val, x_test, y_test, evaluation)

            f = open('{}/logs{}.txt'.format(result_path, args.id), 'a+')
            f.write('\nEpoch: {}\n\t Val_{}: {:.3f} \n\t Test_{}: {:.3f}'.format(
                epoch+1, evaluation, val, evaluation, test))
            f.close()

            if val > best_val:
                best_val = val
                best_test = test
                # histogram(G, D, GAN, x_test, y_test, result_path, latent_dim)
                noise = noise_data(25, latent_dim)
                noise = torch.tensor(noise).float().to(device)
                # show_images(G(noise).cpu().detach().numpy(), result_path)

                if args.save_model:
                    torch.save(G.state_dict(),
                               '{}/gen_anoclass_{}.pth'.format(result_path, ano_class))
                    torch.save(D.state_dict(),
                               '{}/dis_anoclass_{}.pth'.format(result_path, ano_class))

            print("\tGen. Loss: {:.3f}\n\tDisc. Loss: {:.3f}\n\t{}: {:.3f}".format(
                g_loss[-1], d_loss[-1], evaluation, val))
        else:
            print("\tGen. Loss: {:.3f}\n\tDisc. Loss: {:.3f}".format(
                g_loss[-1], d_loss[-1]))

    print('===== End of Adversarial Training =====')
    print('Dataset: {}| Anomalous class: {}| Best test {}: {}'.format(
        args.dataset, ano_class, evaluation, round(best_test, 3)))

    # Saving result in result.json file
    result = [("best_test_{}".format(evaluation), round(best_test, 3)),
              ("best_val_{}".format(evaluation), round(best_val, 3))]
    result_dict = OrderedDict(result)
    with open('{}/result{}.json'.format(result_path, args.id), 'w+') as outfile:
        json.dump(result_dict, outfile, indent=4)


def training_pipeline(args):
    seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)

    x_train, x_test, y_test, x_val, y_val = load_data(args)

    G, D, GAN, dopt, gopt = load_model(args)
    pretrain(args, G, D, GAN, x_train, x_test, y_test, x_val, y_val, dopt)
    train(args, G, D, GAN, x_train, x_test, y_test, x_val, y_val, dopt, gopt)
