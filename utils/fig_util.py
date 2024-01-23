import os

import cv2
import torchvision

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

matplotlib.use('AGG')


def heatmap(vals, fig_path, fig_w=None, fig_h=None, annot=False):
    if fig_w is None:
        fig_w = vals.shape[1]
    if fig_h is None:
        fig_h = vals.shape[0]

    f, ax = plt.subplots(figsize=(fig_w, fig_h), ncols=1)
    sns.heatmap(vals, ax=ax, annot=annot)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def imshow(img, title, fig_path):
    img = torchvision.utils.make_grid(img.cpu().data, normalize=True, nrow=10)
    npimg = img.numpy()
    # fig = plt.figure(figsize=(5, 15))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.title(title)
    # plt.show()

    plt.title(title)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def save_img_by_cv2(img, path):
    img_dir, _ = os.path.split(path)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, img)
