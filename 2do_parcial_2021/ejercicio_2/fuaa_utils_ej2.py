#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 07:50:55 2019

@author: fuaa
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs
from scipy.stats import multivariate_normal as mvn
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.datasets import make_moons
import getpass
import socket
import h5py

plt.rcParams['image.aspect'] = 'equal'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = '#999999'

_THRESHOLD = 1e-8

##################################################################################
def identificar_parcial():
    username = getpass.getuser()
    hostname = socket.gethostname()
    print("Usuario %s en %s." % (username, hostname))


##################################################################################
def printcolor( mensaje, color="k" ):
    """
    Función para imprimir en colores y con el formato de interés.
    """
    if   (color == "r"): mensajeColor = "\x1b[31m" + mensaje + "\x1b[0m"
    elif (color == "g"): mensajeColor = "\x1b[32m" + mensaje + "\x1b[0m"
    elif (color == "y"): mensajeColor = "\x1b[33m" + mensaje + "\x1b[0m"
    elif (color == "b"): mensajeColor = "\x1b[34m" + mensaje + "\x1b[0m"
    elif (color == "p"): mensajeColor = "\x1b[35m" + mensaje + "\x1b[0m"
    elif (color == "c"): mensajeColor = "\x1b[36m" + mensaje + "\x1b[0m"
    else: mensajeColor = mensaje
    mensaje_out = " " + mensajeColor 
    print ( mensaje_out )


##################################################################################
def fuaa_assert(condicion, 
                mensajeFalse = "El resultado no es válido.", 
                mensajeTrue = "Resultado validado." ):
    """
    Función similar al assert.
    """
    # Custom assert.
    if ( condicion ):
        printcolor( mensajeTrue, "g" )
    else:
        printcolor( mensajeFalse, "r" )


##################################################################################
def son_iguales(x1, x2, threshold = _THRESHOLD):
    """
    Evaluar si dos elementos son iguales o no, con una tolerancia dada (threshold).
    """
    if isinstance(x1, float) or isinstance(x1, int):
        dif = np.abs( x1 - x2 )
        condicion = (dif < threshold)
    elif isinstance(x1, np.ndarray):
        dif = np.sqrt(np.sum( ( x1 - x2 )**2 )) / x1.size
        condicion = (dif < threshold)
    else:
        printcolor( "Ningún tipo validado para son_iguales()", "r" )
        condicion = False
    return condicion


##################################################################################
def validar_parametros( parametros, min_params ):
    """
    Validar número de parámetros.
    """
    condicion = (len( parametros ) >= min_params)
    if not condicion:
        printcolor( "[validar_resultado] Insuficientes parámetros (\"%s\"), se necesitan %d, hay %d." % \
            (parametros[0], min_params, len(parametros)), "r" )
    return condicion


##################################################################################
def validar_resultado(*args, **kwargs):
    """"
    Función para validar resultado a invocar desde el notebook.
    """
    # _ES_CORRECTO = False
    _DEBUG = False

    # Abrir el cartel del validación.
    print( "+-------------------------------------------------------------------------+" )
    print( "|               FuAA (2do. parcial 2021): validar resultado               |" )
    print( "+-------------------------------------------------------------------------+" )
    for key, value in kwargs.items():
        if key == "debug":
            _DEBUG = value
    if _DEBUG: 
        print('args:', args)
        print('kwargs:', kwargs)

    if ( len(args) == 0 ):
        print( "| Sin opciones para evaluar.                                              |" )
        print( "+-------------------------------------------------------------------------+" )
        return False


    ###########################################################
    # Ejercicio 2. Evaluar inicializar_mezcla.
    ###########################################################
    elif (args[0] == "inicializar_mezcla"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
        else:
            print("[validar_resultado] Error: llamada sin argumentos (%s)" % args[0])
            return

        N = 7
        d = 2
        k = 2
        X = np.random.randn(N, d)
        semilla = 43
        w, mus, sigmas = test_algoritmo(X, k, semilla)

        fuaa_assert(son_iguales(np.sum(w), 1.), 
            mensajeFalse='La suma de los w debe dar 1.', 
            mensajeTrue="Suma de w_i: validado.")
        fuaa_assert(np.min(w) >= 0, 
            mensajeFalse='Los w deben ser positivos o cero.', 
            mensajeTrue="Todos los w_i son no negativos: validado.")
        fuaa_assert(mus.shape == (k, d), 
            mensajeFalse='El tamaño de la matriz de mus no es correcto.', 
            mensajeTrue="Tamaño de matriz de mus: validado.")
        for j in range(k):
            fuaa_assert(np.allclose(sigmas[j], np.eye(len(sigmas[j]))), 
                mensajeFalse='Los sigmas deben inicializarse a la identidad.', 
                mensajeTrue="Matriz Sigma es identidad: validado.")


    ###########################################################
    # Ejercicio 2. Evaluar expectation_step.
    ###########################################################
    elif (args[0] == "expectation_step"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
        else:
            print("[validar_resultado] Error: llamada sin argumentos (%s)" % args[0])
            return

        np.random.seed(3)
        N = 2
        d = 3
        k = 2
        X_ = np.random.randn(N, d)
        w_ = np.random.rand(k)
        w_ = w_ / np.sum(w_)
        mus_ = np.random.randn(k, d)
        sigmas_ = np.random.randn(k, d, d)
        for j in range(k): 
            sigmas_[j] = sigmas_[j] @ sigmas_[j].T
        gammas = test_algoritmo(X_, w_, mus_, sigmas_)
        gammas_correctos = np.array([[1.34208238e-04, 9.99865792e-01],
                                    [9.99999062e-01, 9.38144350e-07]])
        fuaa_assert(son_iguales(gammas, gammas_correctos),
            mensajeFalse="Probabilidades de pertenencia a cada cluster no validadas.",
            mensajeTrue="Probabilidades de pertenencia a cada cluster validadas.")


    ###########################################################
    # Ejercicio 2. Evaluar maximization_step
    ###########################################################
    elif (args[0] == "maximization_step"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
        else:
            print("[validar_resultado] Error: llamada sin argumentos (%s)" % args[0])
            return

        np.random.seed(84)
        N = 5
        d = 2
        k = 2
        X_ = np.random.randn(N, d)
        gammas_ = np.random.randn(N, k)
        w, mus, sigmas = test_algoritmo(X_, gammas_)

        w_correcto = np.array([-0.50647492, 0.60709566])
        mus_correctos = np.array([[ 0.03196345, -1.57011573], [-0.12383003, -0.67268656]])
        sigmas_correctos = np.array([[[ 0.58728466, -0.36449661],[-0.36449661,  0.4157087 ]],
                                    [[ 0.21543145, -0.16303434],[-0.16303434,  0.30495924]]])

        fuaa_assert(son_iguales(w, w_correcto), 
            mensajeFalse="Vector de pesos de la mezcla: no validado.", 
            mensajeTrue="Vector de pesos de la mezcla: validado.")
        fuaa_assert(son_iguales(mus, mus_correctos), 
            mensajeFalse="Medias en los cluster: no validadas.", 
            mensajeTrue="Medias en los cluster: validadas.")
        fuaa_assert(son_iguales(sigmas, sigmas_correctos), 
            mensajeFalse="Matrices de covarianza: no validadas.", 
            mensajeTrue="Matrices de covarianza: validadas.")


    ###########################################################
    # Test.
    ###########################################################
    elif (args[0] == "test"):
        if validar_parametros( args, 4 ):
            condicion = args[1]
            mensajeF = args[2]
            mensajeT = args[3]
            fuaa_assert( condicion, mensajeFalse = mensajeF, mensajeTrue = mensajeT )

    ###########################################################
    # No hay ninguna opción de ejercicio.
    ###########################################################
    else:
        printcolor( "Ninguna opción revisada." ) 

    # Cerrar el cartel.
    print( "+-------------------------------------------------------------------------+" )



###########################################################
# Generar datos
###########################################################
def generar_datos():
    X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.50, random_state=0)
    X = X[:, ::-1] # flip axes for better plotting
    np.random.seed(13)
    Xalt = np.dot(X, np.random.randn(2, 2))
    return Xalt

###########################################################
# Plot scatter points
###########################################################
def plot_scatter(x, y, c=None, s=50, cmap=None):
    plt.scatter(x, y, c=c, alpha=0.5, s=s, cmap=cmap)
    plt.axis('scaled');


###########################################################
# Draw ellipses
###########################################################
def draw_ellipse(position, covariance, ax=None, **kwargs):
    '''
    Dibujar una elipse en un posición y con una covarianza dada.
    '''
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


###########################################################
# Evaluar una gaussiana multivariada en un punto
###########################################################
def evaluar_kernel_gaussiano(x, mu, Sigma):
    '''
    Entrada:
        x: vector a evaluar de dimensión (d,1)
        mu: media del núcleo gaussiano de dimensión (d,1)
        Sigma: covarianza del núcleo gaussiano de dimensión (d,d)
    Salida:
        p: resultado de evaluar el kernel gaussiano
    '''
    d = len(mu)
    det_sigma = np.linalg.det(Sigma)
    normalization_factor = np.power(2 * np.pi, d / 2) * np.sqrt(det_sigma)
    mahalanobis_distance = (x - mu).T @ np.linalg.inv(Sigma) @ (x - mu)
    p = 1.0 / normalization_factor * np.exp(-0.5 * mahalanobis_distance)
    p = np.squeeze(p)  # para asegurar que la salida es un escalar
    return p


###########################################################
# Calcular la log verosimilitud 
###########################################################
def log_verosimilitud(X, w, mus, sigmas):
    '''
    Entrada:
        X: matriz de tamaño Nxd que contiene las muestras.
        w: arreglo de tamaño k que contiene los pesos actuales.
        mus: arreglo de tamaño (k,d) que contiene las medias, una por fila.
        sigmas: arreglo de tamaño (k,d,d) que contiene las matrices de covarianza.
     Salida:
        log_ver: logaritmo de la verosimilitud de las muestras con el modelo.
    '''
    N = X.shape[0]
    k = len(w)
    log_ver = 0
    for i in range(N):
        s = 0
        for j in range(k):
            s += w[j] * mvn(mus[j], sigmas[j]).pdf(X[i])
            #s += w[j] *  evaluar_kernel_gaussiano(X[i], mus[j], sigmas[j])
        log_ver += np.log(s)
    return log_ver


###########################################################
# Plot GMM
###########################################################
def plot_gmm(X, gmm, labels_gmm, cmap=None):
    plot_scatter(X[:, 0], X[:, 1], c=labels_gmm, cmap=cmap)
    w_factor = 0.1 / gmm['weights'].max()
    for pos, covar, w in zip(gmm['means'], gmm['covars'], gmm['weights']):
        # draw_ellipse(pos, covar, alpha=w * w_factor, edgecolor='black' )
        draw_ellipse(pos, covar, facecolor='none', edgecolor='cornflowerblue', alpha=0.5)


###########################################################
# Mostrar 100 dígitos
###########################################################
def plot_digits(data, title=''):
    fig, ax = plt.subplots(10, 10, figsize=(8, 8),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)
    fig.suptitle(title, fontsize=16,y=0.93)


###########################################################
# Plot generar_muestras_gaussiana_multivariada
###########################################################
def generar_muestras_gaussiana_multivariada(mean_, Sigma_, N=100):
    '''
    Función para generar N nuestras de una Gaussiana multivariada de dimensión
    d usando la descomposición de Cholesky.
    
    Entrada: 
        mean_: Arreglo de tamaño (1,d) que contiene la media.
        Sigma_: Arreglo de tamaño (dxd) que contiene la matriz de covarianza.
        n: Número de puntos a generar.
        
    Salida: 
        X: Arreglo de tamaño (Nxd).
    '''
    d = mean_.shape[0]
    d1, d2 = Sigma_.shape
    if not (d == d1 == d2):
        print('Error de dimensiones.')
    epsilon = 1e-8
    Sigma = Sigma_ + epsilon * np.identity(d)
    L = np.linalg.cholesky(Sigma)
    u = np.random.normal(loc=0, scale=1, size=d * N).reshape(d, N)
    X = mean_ + np.dot(L, u).T
    return X

###########################################################
# Inicializar mezcla.
###########################################################
def inicializar_mezcla(X, Ng, semilla):
    '''
    Entrada:
        X: matriz de tamaño (N,d) que contiene N muestras, una por fila.
        Ng: número de clusters a encontrar.
    Salida:
        w: arreglo de largo Ng que contiene los pesos de la mezcla. 
           Se deben inicializar a valores aleatorios cuya suma sea 1.
        mus: arreglo de tamaño (Ng,d) que contiene los pesos.
        sigmas: arreglo de tamaño (Ng,d,d) que contiene las matrices de 
                covarianza de los clusters
    '''
    N, d = X.shape
    np.random.seed(semilla)
    indices = np.random.permutation(X.shape[0])[:Ng]
    mus = X[indices]
    sigmas = np.array([np.eye(d)] * Ng)
    w = np.random.random(Ng)
    w /= w.sum()
    return w, mus, sigmas


###########################################################
# Maximization step
###########################################################
def maximization_step(X, gammas):
    '''
    Entrada:
        X: matriz de tamaño Nxd con las muestras a evaluar.
        gammas: arreglo de tamaño (N,k) con las probabilidades de pertenencia 
                a cada cluster.
        
    Salida:
        w: vector de pesos de la mezcla.
        mus: arreglo de tamaño (k,d) que contiene las medias en el paso 
             actual.
        sigmas: arreglo de tamaño (k,d,d) que contiene las matrices de 
                covarianza de los clusters en el paso actual.    
    '''
    N, d = X.shape
    N, k = gammas.shape
    w = np.mean(gammas, axis=0)
    mus = np.zeros((k, d))
    for j in range(k):
        for i in range(N):
            mus[j] += gammas[i, j] * X[i]
        mus[j] /= gammas[:, j].sum()
    sigmas = np.zeros((k, d, d))
    for j in range(k):
        for i in range(N):
            delta = np.reshape(X[i] - mus[j], (d, 1))
            sigmas[j] += gammas[i, j] * np.dot(delta, delta.T)
        sigmas[j] /= gammas[:, j].sum()
    return w, mus, sigmas