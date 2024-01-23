import argparse
import pickle
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('AGG')


def rgb(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--feature_dir', default='', type=str, help='feature dir')
    parser.add_argument('--save_dir', default='', type=str, help='save dir')
    args = parser.parse_args()

    data0_path = os.path.join(args.feature_dir, 'layer{}_nor_1000_a.pkl')
    data1_path = os.path.join(args.feature_dir, 'layer{}_adv_1000_a.pkl')
    datam_path = os.path.join(args.feature_dir, 'layer{}_nor_50_a.pkl')
    save_path = os.path.join(args.save_dir, 'layer{}_{}.png')

    layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    labels = [0]

    # ###############################
    # # feature distribution
    # ###############################
    # for layer in layers:
    #     print('===>layer', layer)
    #
    #     data0_file = open(data0_path.format(layer), 'rb')
    #     data0 = pickle.load(data0_file)  # [C, N, D]
    #     data1_file = open(data1_path.format(layer), 'rb')
    #     data1 = pickle.load(data1_file)  # [C, N, D]
    #     print(data0.shape)
    #     print(data1.shape)
    #
    #     data0 = data0[:, :, 0:10]  # [C, N, D]
    #     data1 = data1[:, :, 0:10]
    #
    #     for label in labels:
    #         print('===>label', layer)
    #         data0_ = data0[label]  # [N, D]
    #         data1_ = data1[label]  # [N, D]
    #         data = np.stack((data0_, data1_), axis=0)  # [2, N, D]
    #
    #         ####################VISUALIZE DATA####################
    #         fig_path = save_path.format(layer, 'DIS')
    #         print(fig_path)
    #         T, N, D = data.shape
    #
    #         x = np.tile(np.arange(0, D), N * T)  # [0,1,n 0,1,n 0,1,n]
    #         z = np.repeat(np.arange(0, T), N * D)  # [0,0,0 1,1,1]
    #         y = data.flatten()
    #
    #         plt.figure(figsize=(50, 10))
    #         sns.set(font_scale=2, style="white")
    #
    #         palette = {0: rgb(154, 202, 124), 1: rgb(141, 181, 224)}
    #         v = sns.violinplot(x=x, y=y, hue=z, palette=palette, density_norm='count')
    #         v.legend_.remove()
    #
    #         plt.savefig(fig_path.format(layer), bbox_inches='tight')
    #         plt.clf()
    #         ####################VISUALIZE DATA####################

    ###############################
    # feature pattern
    ###############################
    for layer in layers:
        print('===>layer', layer)

        data0_file = open(data0_path.format(layer), 'rb')
        data0 = pickle.load(data0_file)  # [C, N, D]
        data1_file = open(data1_path.format(layer), 'rb')
        data1 = pickle.load(data1_file)  # [C, N, D]
        datam_file = open(datam_path.format(layer), 'rb')
        datam = pickle.load(datam_file)  # [C, N, D]

        data0_mean = np.mean(data0, axis=2, keepdims=True)  # (c, n, 1)
        data0_std = np.std(data0, axis=2, keepdims=True)  # (c, n, 1)
        data0 = (data0 - data0_mean) / (data0_std + 1e-5)
        data1_mean = np.mean(data1, axis=2, keepdims=True)  # (c, n, 1)
        data1_std = np.std(data1, axis=2, keepdims=True)  # (c, n, 1)
        data1 = (data1 - data1_mean) / (data1_std + 1e-5)
        datam_mean = np.mean(datam, axis=2, keepdims=True)  # (c, n, 1)
        datam_std = np.std(datam, axis=2, keepdims=True)  # (c, n, 1)
        datam = (datam - datam_mean) / (datam_std + 1e-5)

        data0 = data0[:, :, 0:10]
        data1 = data1[:, :, 0:10]
        datam = datam[:, :, 0:10]
        datam = np.mean(datam, axis=1)  # (c, n, d) -> (c, d)

        for label in labels:
            print('===>label', layer)
            data0_ = data0[label]  # [N, D]
            data1_ = data1[label]  # [N, D]
            datam = datam[label]
            print(data0_.shape, data1_.shape)

            ####################VISUALIZE DATA####################
            fig_path = save_path.format(layer, 'PAT')
            print(fig_path)

            plt.figure(figsize=(50, 10))
            sns.set(font_scale=2, style="white")

            N, D = data1_.shape
            X = np.tile(np.arange(D), N)  # [0,1,n 0,1,n 0,1,n]

            C1 = rgb(96, 160, 56)
            C2 = rgb(75, 135, 203)
            C3 = rgb(98, 93, 94)

            sns.lineplot(x=X, y=data0_.flatten(), errorbar=("sd"),
                         marker='_', markersize='80', markeredgewidth=10, markeredgecolor=C1,
                         color=C1, linewidth=7)
            sns.lineplot(x=X, y=data1_.flatten(), errorbar=("sd"),
                         marker='_', markersize='80', markeredgewidth=10, markeredgecolor=C2,
                         color=C2, linewidth=7)

            plt.plot(np.arange(D), datam,
                     marker='_', markersize='80', markeredgewidth=5,
                     color=C3, linewidth=7, linestyle='--')

            plt.savefig(fig_path.format(layer), bbox_inches='tight')
            plt.clf()
            ####################VISUALIZE DATA####################


if __name__ == '__main__':
    main()
