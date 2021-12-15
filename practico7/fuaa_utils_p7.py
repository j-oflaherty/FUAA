#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 07:52:52 2021

@author: fuaa
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import h5py
import time
import os
import glob


_THRESHOLD = 1e-8
_FIGSIZE = (10, 10)


###############################################################################
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


###############################################################################
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


###############################################################################
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


##########################################################################
def validar_parametros( parametros, min_params ):
    """
    Validar número de parámetros.
    """
    condicion = (len( parametros ) >= min_params)
    if not condicion:
        printcolor( "[validar_resultado] Insuficientes parámetros (\"%s\"), se necesitan %d, hay %d." % \
            (parametros[0], min_params, len(parametros)), "r" )
    return condicion
    

#########################################################################################
def validar_resultado(*args, **kwargs):
    """
    Función para validar resultado a invocar desde el notebook.
    """
    # _ES_CORRECTO = False
    _DEBUG = False

    # Abrir el cartel del validación.
    print("+-------------------------------------------------------------------------+")
    print("|                         FuAA: validar resultado                         |")
    print("+-------------------------------------------------------------------------+")
    for key, value in kwargs.items():
        if key == "debug":
            _DEBUG = value
    if _DEBUG:
        print('args:', args)
        print('kwargs:', kwargs)

    if (len(args) == 0):
        print("| Sin opciones para evaluar.                                              |")
        print("+-------------------------------------------------------------------------+")
        return False


    ###########################################################
    # Práctico 7. Split data.
    ###########################################################
    elif (args[0] == "train_valid_test"):
        if validar_parametros( args, 3 ):
            X_train = args[1]
            X_valid = args[2]
            X_test = args[3]
            N_train = X_train.shape[0]
            N_valid = X_valid.shape[0]
            N_test = X_test.shape[0]
            
            condicion1 = np.ceil(N_train/N_valid) == 7
            condicion2 = np.ceil((N_train + N_valid)/N_test) == 4
            fuaa_assert( condicion1, 
            mensajeFalse='Relación de tamaños de X_train y X_valid no es válida.',
            mensajeTrue='Relación de tamaños de X_train y X_valid validada.' )
            fuaa_assert( condicion2, 
            mensajeFalse='Relación de tamaños de X_train, X_valid y X_test no es válida.',
            mensajeTrue='Relación de tamaños de X_train, X_valid y X_test validada.' )


    ###########################################################
    # Práctico 7. Evaluar kernel gaussiano.
    ###########################################################
    elif (args[0] == "evaluar_kernel_gaussiano"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
        else:
            print("[validar_resultado] Error: llamada sin argumentos (%s)" % args[0])
            return

        np.random.seed(33)
        d = 5
        x_ = np.random.randn(d, 1)
        mu_ = np.random.randn(d, 1)
        Sigma_ = np.random.randn(d, d)
        Sigma_ = Sigma_ @ Sigma_.T  # para que sea semidefinida positiva
        p = test_algoritmo(x_, mu_, Sigma_)
        p_ = 2.29350214e-07
        fuaa_assert(son_iguales(p, p_), 
            mensajeFalse="Resultado de kernel gaussiano no validado.",
            mensajeTrue="Resultado de kernel gaussiano validado." )


    ###########################################################
    # Práctico 7. Evaluar estimar_densidad.
    ###########################################################
    elif (args[0] == "estimar_densidad"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
        else:
            print("[validar_resultado] Error: llamada sin argumentos (%s)" % args[0])
            return

        np.random.seed(45)
        N = 4
        M = 6
        d = 3
        X_ = np.random.randn(N, d)
        Z_ = np.random.randn(M, d)
        Sigma_ = np.random.randn(d, d)
        Sigma_ = Sigma_ @ Sigma_.T  # para que sea semidefinida positiva
        densidades = test_algoritmo(X_, Z_, Sigma_)
        densidades_ = np.array([5.98438614e-02, 0, 4.83415432e-02, 1.86978536e-20])

        fuaa_assert(son_iguales(densidades, densidades_), 
            mensajeFalse="Resultado de densidades estimadas no validado.",
            mensajeTrue="Resultado de densidades estimadas validado." )


    ###########################################################
    # Práctico 7. Evaluar inicializar_mezcla.
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

        # sigmas_correctos = np.array([[[1., 0.], [0., 1.]], [[1., 0.], [0., 1.]]])

        fuaa_assert(np.sum(w) == 1, 
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
    # Práctico 7. Evaluar expectation_step.
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
    # Práctico 7. Evaluar maximization_step
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
    # Práctico 7. Evaluar log_verosimilitud.
    ###########################################################
    elif (args[0] == "log_verosimilitud"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
        else:
            print("[validar_resultado] Error: llamada sin argumentos (%s)" % args[0])
            return

        np.random.seed(22)
        N = 3
        d = 3
        k = 2
        X_ = np.random.randn(N, d)
        w_ = np.random.rand(k)
        w_ = w_ / np.sum(w_)
        mus_ = np.random.randn(k, d)
        sigmas_ = np.random.randn(k, d, d)
        for j in range(k):
            sigmas_[j] = sigmas_[j] @ sigmas_[j].T
        log_ver = test_algoritmo(X_, w_, mus_, sigmas_)
        log_ver_ = -28.48357133785

        fuaa_assert(son_iguales(log_ver, log_ver_), 
            mensajeFalse="Verosimilitud no validada.",
            mensajeTrue="Verosimilitud validada.",)
        

    #####################################################################
    # No hay ninguna opción de ejercicio.
    #####################################################################
    else:
        printcolor("Ninguna opción revisada.")

    # Cerrar el cartel.
    print("+-------------------------------------------------------------------------+")

# condicion = False
# mensaje = "Este ese el texto a mostrar en caso de condición falsa."
# validar_resultado( "test", condicion, mensaje )

###############################################################################
def error_relativo(x, y):
    ''' devuelve el error relativo'''
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))