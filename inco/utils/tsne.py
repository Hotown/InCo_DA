from cProfile import label

import matplotlib
import torch

matplotlib.use('Agg')
import matplotlib.colors as col
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,
              filename: str, source_color='r', target_color='b'):
    """
    Visualize features from different domains using t-SNE.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'

    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # domain labels, 1 represents source while 0 represents target
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

    # visualize using matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    font1 = {
        'family' : 'Times New Roman',
        'size': 18
    }
    plt.scatter(X_tsne[:len(source_feature), 0], X_tsne[:len(source_feature), 1], s=20, label='$source$', c=source_color)
    plt.legend(prop=font1)
    plt.scatter(X_tsne[len(source_feature):, 0], X_tsne[len(source_feature):, 1], s=20, label='$target$', c=target_color)
    plt.legend(prop=font1)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename)
