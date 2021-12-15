#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 07:52:52 2019

@author: fuaa
"""
import numpy as np

def entrenar_perceptron(X, y, w_inicial=None, max_iter = 500):
    """
    Entrada:
        X: matriz de (Nxd+1) que contiene las muestras de entrenamiento
        y: etiquetas asociadas a las muestras de entrenamiento
        max_iter: máxima cantidad de iteraciones que el algoritmo 
                        puede estar iterando
        w_inicial: inicialización de los pesos del perceptrón
        
    Salida:
        w: parámetros del modelos perceptrón   
        error: lista que contiene los errores cometidos en cada iteración
    """
    
    if w_inicial is None:
        # Se inicializan los pesos del perceptrón
        w = np.random.rand(X.shape[1]) # w = np.zeros(d+1)
    else:
        w = w_inicial

    #######################################################
    ######## EMPIEZA ESPACIO PARA COMPLETAR CODIGO ########
    #######################################################
    
    n_iter = 0    
    error = []
    N = X.shape[0]
    hay_muestras_mal_clasificadas = True
    while ((n_iter < max_iter) and hay_muestras_mal_clasificadas):
        
        # se calcula el score
        score = np.dot(X, w)   
        
        # se encuentran las muestras mal clasificadas
        indices_mal_clasificados = y != np.sign(score) 
        
        # se calcula el error y se lo almacena
        cantidad_mal_clasificadas = np.sum(indices_mal_clasificados) 
        error_i = cantidad_mal_clasificadas/N
        
        error.append(error_i)
        
        # actualización de los pesos
        if error_i == 0:
            hay_muestras_mal_clasificadas = False     
        else:    
            # si el error es mayor que cero se elige una de las muestras 
            # mal clasificadas para actualizar los pesos 
            # (se eligió la primera pero podría ser cualquiera de ellas)
            w = w + y[indices_mal_clasificados][0] * X[indices_mal_clasificados][0]  # se actualizan los pesos

        n_iter = n_iter+1
    
    #######################################################
    ######## TERMINA ESPACIO PARA COMPLETAR CODIGO ########
    #######################################################
    
    return w, error


def generar_semianillos_validacion(N, radio, ancho, separacion, semilla = None):
    '''
    Entrada:
        N: número de muestras a generar
        radio: radio interior del semicírculo
        ancho: diferencia entre el radio exterior e interior
        separación: separación entre los semicírculos
        semilla: valor que se le asigna al método random.seed()

    Salida:
        X: matríz de Nx3 que contiene los datos generados en coordenadas homogéneas
        y: estiquetas asociadas a los datos
    '''
        
    if semilla is not None:
        np.random.seed(semilla)
        
    X = np.ones((N, 3))
    # se sortea a que clase pertenecen las muestras
    y = 2 * (np.random.rand(N) < 0.5) - 1
    
    # radios y ángulos del semicírculo superior
    radios = radio + ancho * np.random.rand(N)
    thetas = np.pi * np.random.rand(N)
    # coordenadas en x de ambos semicírculos
    X[:,1] = radios * np.cos(thetas) * y + (radio + ancho/2)*(y==-1)
    # coordenadas en y de ambos semicírculos
    X[:,2] = radios * np.sin(thetas) * y - separacion * (y==-1)
    
    return X, y


#####################################################################################
