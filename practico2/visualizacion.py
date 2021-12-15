#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 07:50:55 2019

@author: fuaa
"""
from matplotlib import pyplot as plt
import numpy as np


def visualizar_conjunto_entrenamiento(X, y):

    '''
    Entrada: 
        X: matriz de tamaño Nx3 que contiene el conjunto de puntos de entrenamiento expresados en
           en coordenadas homogeneas. La primera coordenada de cada punto es uno.
        y: etiquetas asignadas a los puntos
    '''
    plt.figure(figsize=(8,8))

    # Se grafican los puntos sorteados
    plt.scatter(X[y==-1, 1],X[y==-1, 2], s=40, color='r', marker='*', label='etiqueta -1')
    plt.scatter(X[y==1, 1], X[y==1, 2], s=40, color='b', marker='o', label='etiqueta 1')

    plt.legend()       
    plt.axis('equal')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Conjunto de entrenamiento generado')



##################################################################################

def visualizar_modelo_lineal(X, y, w_g):

    '''
    Entrada: 
        X: matriz de tamaño Nx3 que contiene el conjunto de puntos de entrenamiento expresados en
           en coordenadas homogeneas. La primera coordenada de cada punto es uno.
        y: etiquetas asignadas a los puntos
        w_g: parámetros del modelo lineal encontrados 
    '''
    plt.figure(figsize=(8,8))

    # Se grafican los puntos sorteados
    plt.scatter(X[y==-1, 1],X[y==-1, 2], s=40, color='r', marker='*', label='etiqueta -1')
    plt.scatter(X[y==1, 1], X[y==1, 2], s=40, color='b', marker='o', label='etiqueta 1')
    
    x1_min = X[:,1].min() 
    x1_max = X[:,1].max()
    x1 = np.linspace(x1_min , x1_max)
    if w_g[2]==0:
        x2_g = -(w_g[0]/w_g[2])*np.ones(x1.shape)
    else:
        # Se grafica la superficie de decisión encontrada
        x2_g = - w_g[1]/w_g[2] * x1 + -w_g[0]/w_g[2]
    plt.plot(x1, x2_g, label = 'funcion encontrada')

    plt.legend()       
    plt.axis('equal')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Modelo lineal encontrado')
    
###############################################################################


