import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_curve
import torch


def deprocess(x):
    x = (x+1)/2 * 255
    x = np.clip(x, 0, 255)
    x = np.uint8(x)
    return x


def show_images(img_array, result_path):
    # if dataset == 'mnist':
    n_images = deprocess(img_array[:25])

    plt.figure(figsize=(5, 5))
    for i in range(len(n_images)):
        img = np.squeeze(n_images[i, ...])
        plt.subplot(5, 5, i+1)
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 0, 1)
            plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig('{}/generated_data_at_best_epoch.png'.format(result_path))
    plt.close()


def compute_au(D, G, GAN, x_val, y_val, x_test, y_test, mode):
    '''
    Return auprc or auroc evaluated on validation/test set
    '''
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    D = D.to(device)
    x_val = torch.tensor(x_val).to(device).float()
    x_test = torch.tensor(x_test).to(device).float()

    if mode == 'auprc':
        # VALIDATION
        # y_pred_val = np.squeeze(D(x_val))
        y_pred_val = D(x_val).cpu().detach().numpy()
        precision, recall, _ = precision_recall_curve(y_val, y_pred_val)
        val_prc = auc(recall, precision)
        # TEST
        # y_pred_test = np.squeeze(D(x_test))
        y_pred_test = D(x_test).cpu().detach().numpy()
        precision, recall, _ = precision_recall_curve(y_test, y_pred_test)
        test_prc = auc(recall, precision)

        return val_prc, test_prc

    elif mode == 'auroc':
        # VALIDATION
        # y_pred_val = np.squeeze(D(x_val))
        y_pred_val = D(x_val).cpu().detach().numpy()
        fpr, tpr, _ = roc_curve(y_val, y_pred_val)
        val_roc = auc(fpr, tpr)
        # TEST
        # y_pred_test = np.squeeze(D(x_test))
        y_pred_test = D(x_test).cpu().detach().numpy()
        fpr, tpr, _ = roc_curve(y_test, y_pred_test)
        test_roc = auc(fpr, tpr)

        return val_roc, test_roc


def histogram(G, D, GAN, x_test, y_test, result_path, latent_dim):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    D = D.to(device)
    x_test = torch.tensor(x_test).to(device).float()

    gan_in = torch.tensor(np.random.normal(
        0, 1, [5000, latent_dim])).to(device)
    # y_gen_pred = np.squeeze(gan_in)

    y_gen_pred = gan_in
    y_pred = D(x_test)

    y_gen_pred = y_gen_pred.cpu().detach().numpy()
    y_gen_pred = np.squeeze(y_gen_pred)
    print(y_gen_pred.shape)
    y_gen_pred = y_gen_pred.flatten()
    print(y_gen_pred.shape)
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = np.squeeze(y_pred)
    x_test = x_test.cpu().detach().numpy()

    plt.figure()
    # 1 is normal data
    # 0 is anomalous data
    plt.hist(y_pred[np.where(y_test == 0)], density=True, bins=100,
             range=(0, 1.0), label='anomalous', color='r', alpha=0.5)
    plt.hist(y_pred[np.where(y_test == 1)], density=True, bins=100,
             range=(0, 1.0), label='normal', color='b', alpha=0.5)
    plt.hist(y_gen_pred, density=True, bins=100, range=(
        0, 1.0), label='generated', color='g', alpha=0.5)

    #plt.axis([0, 1, 0, 1])
    plt.xlabel('Discriminator Score')
    plt.ylabel('Frequency of Occurence')
    plt.title('Histogram of Discriminator Score at the Best Epoch')
    plt.legend(loc=9)
    plt.savefig('{}/histogram_at_best_epoch.png'.format(result_path), dpi=60)
    plt.close()
