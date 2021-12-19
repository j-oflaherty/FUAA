#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 07:50:55 2019

@author: fuaa
"""
from matplotlib import pyplot as plt
import numpy as np
import getpass
import socket
import itertools


##################################################################################
def identificar_parcial():
    username = getpass.getuser()
    hostname = socket.gethostname()
    print("Usuario %s en %s." % (username, hostname))


##################################################################################
def plot_svc_decision_function(model, ax=None, color='black'):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors=color,
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
            
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

##################################################################################

def fashion_mnist_load_data(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load fashion MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


##################################################################################

def fashion_mnist_plot_data(X, y, y_pred=[], num_row=5, num_col=8):
    """Plot fashion MNIST data """

    # labels as strings
    labs_txt = ['T-shirt/top', 'Trowser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
   
    # se obtienen los datos
    num = num_row * num_col
    images = X[:num]
    labels = y[:num]
    if np.asarray(y_pred).shape == (0,):
        labels_pred = labels
    else:
        labels_pred = y_pred[:num]

    # se grafican las imágenes
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i in range(num_row*num_col):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(images[i].reshape(28,28), cmap='gray')
        ax.set_title('{}'.format(labs_txt[labels[i]]),
                     color='black' if labels[i] == labels_pred[i] else 'red')
        
    plt.tight_layout()


##################################################################################
def fuaa_plot_confusion_matrix(cm, classes,
                               normalize=False,
                               title='Confusion matrix',
                               cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')