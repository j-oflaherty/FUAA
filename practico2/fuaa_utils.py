#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 07:52:52 2021

@author: fuaa
"""
import numpy as np
import h5py
from algoritmos import generar_semianillos_validacion

_THRESHOLD = 10e-9

"""
Función para imprimir en colores y con el formato de interés.
"""
def printcolor( mensaje, color="k" ):
    if   (color == "r"): mensajeColor = "\x1b[31m" + mensaje + "\x1b[0m"
    elif (color == "g"): mensajeColor = "\x1b[32m" + mensaje + "\x1b[0m"
    elif (color == "y"): mensajeColor = "\x1b[33m" + mensaje + "\x1b[0m"
    elif (color == "b"): mensajeColor = "\x1b[34m" + mensaje + "\x1b[0m"
    elif (color == "p"): mensajeColor = "\x1b[35m" + mensaje + "\x1b[0m"
    elif (color == "c"): mensajeColor = "\x1b[36m" + mensaje + "\x1b[0m"
    else: mensajeColor = mensaje
    mensaje_out = " " + mensajeColor 
    print ( mensaje_out )

"""
Función similar al assert.
"""
def fuaa_assert(   condicion, 
                mensajeFalse = "El resultado no es válido.", 
                mensajeTrue = "Resultado validado." ):

    # Custom assert.
    if ( condicion ):
        printcolor( mensajeTrue, "g" )
    else:
        printcolor( mensajeFalse, "r" )
    
    # Assert tradicional
    # assert condicion, mensajeFalse
    
    return

"""
Evaluar si dos elementos son iguales o no, con una tolerancia dada (threshold).
"""
def son_iguales(x1, x2, threshold = _THRESHOLD):
    if x1.shape == x2.shape:
        if isinstance(x1, np.ndarray):
            dif = np.sqrt(np.sum( ( x1 - x2 )**2 )) / x1.size
        elif isinstance(x1, float):
            dif = np.abs( x1 - x2 )
        condicion = (dif < threshold)
    else:
        condicion = False
    return condicion

"""
Validar número de parámetros.
"""
def validar_parametros( parametros, min_params ):
    condicion = (len( parametros ) >= min_params)
    if not condicion:
        printcolor( "[validar_resultado] Insuficientes parámetros (\"%s\"), se necesitan %d,  hay %d." % \
            (parametros[0], min_params, len(parametros)), "r" )
    return condicion


""""
Función para validar resultado a invocar desde el notebook.
"""
def validar_resultado(*args, **kwargs):
    # _ES_CORRECTO = False
    _DEBUG = False

    # Abrir el cartel del validación.
    print( "+-------------------------------------------------------------------------+" )
    print( "|                         FuAA: validar resultado                         |" )
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
    # Práctico 2. Ejercicio 2c (pocket).
    ###########################################################
    elif (args[0] == "pocket"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
        else:
            print( "[validar_resultado] Error: llamada sin argumentos (%s)" % args[0] )
            return

        # Generar semianillos linealmente separables y ejecutar el algoritmo pocket implementado.
        N = 2000
        radio = 10
        ancho = 5
        separacion = 5
        semilla = 42
        X, y =  generar_semianillos_validacion(N, radio, ancho, separacion, semilla = semilla)
        w_inicial = np.zeros(X.shape[1])
        num_iteraciones = 100000
        w_pocket, error_entrenamiento = test_algoritmo(X, y, w0 = w_inicial, max_iter = num_iteraciones)

        # Validar error cero con datos linealmente separables.
        error_pocket = np.asarray( error_entrenamiento )
        condicion2 = ( error_pocket[-1] == 0)
        fuaa_assert(   condicion2, \
                    mensajeFalse="El error del algoritmo Pocket debe ser cero para datos linealmente\n separables. Revisar algoritmo.", \
                    mensajeTrue="Error nulo con datos linealmente separables: resultado validado.")  

        # Validar error monótono decreciente con datos linealmente separables.
        condicion3 = sum( error_pocket[:-1] - error_pocket[1:] <= 0)
        fuaa_assert(   condicion3, \
                    mensajeFalse="El error del algoritmo Pocket debe ser monótono decreciente.\n Revisar algoritmo", \
                    mensajeTrue="Error monótono decreciente: resultado validado.") 

    ###########################################################
    # Práctico 2. Transformación polinomio tercer grado.
    ###########################################################
    elif (args[0] == "transformacion"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
        else:
            print( "[validar_resultado] Error: llamada sin argumentos (%s)" % args[0] )
            return

        pts = np.array([[1, 1, 1], [1, 2, 3], [1, 0, 2], [1, 2, 0] ])
        pts_t_true = np.array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
                            [ 1.,  2.,  3.,  4.,  6.,  9.,  8., 12., 18., 27.],
                            [ 1.,  0.,  2.,  0.,  0.,  4.,  0.,  0.,  0.,  8.],
                            [ 1.,  2.,  0.,  4.,  0.,  0.,  8.,  0.,  0.,  0.]])
        pts_t = test_algoritmo( pts )

        # Validar dimensión de datos transformados.
        condicion1 = pts_t.shape == pts_t_true.shape
        fuaa_assert(   condicion1, \
                    mensajeFalse = "La dimensión de los datos transformados no es correcta.", \
                    mensajeTrue= "Dimensión de salida: resultado validado.")

        # Validar transformación.
        condicion2 = son_iguales( pts_t_true, pts_t )
        fuaa_assert(   condicion2, \
                    mensajeFalse = "La transformación implementada no es correcta.", 
                    mensajeTrue = "Transformación: resultado validado.")

    ###########################################################
    # Práctico 2. Características de los dígitos.
    ###########################################################
    elif (args[0] == "digitos"):
        if validar_parametros( args, 3 ):
            feat_train = args[1]
            feat_test = args[2]

            condicion1 = ( feat_train.shape == (1561, 2) )
            fuaa_assert(    condicion1, \
                            mensajeFalse = "Dimensiones de las características de entrenamiento: no son correctas.", \
                            mensajeTrue = "Dimensiones de las características de entrenamiento: resultado validado." )
 
            condicion2 = ( feat_test.shape == (424, 2) )
            fuaa_assert(    condicion2, \
                            mensajeFalse = "Dimensiones de las características de test: no son correctas.", \
                            mensajeTrue = "Dimensiones de las características de test: resultado validado." )

            # Leer datos correctos para comparar.
            path='usps/features_digitos.h5'
            with h5py.File(path, 'r') as hf:
                featuresTrain = hf.get('featuresTrain')[:]
                featuresTest = hf.get('featuresTest')[:]

            condicion3 = son_iguales( feat_train, featuresTrain )
            fuaa_assert(    condicion3, \
                            mensajeFalse = "Características de entrenamiento: no son correctas.", \
                            mensajeTrue = "Características de entrenamiento: resultado validado." )
            
            condicion4 = son_iguales( feat_test, featuresTest )
            fuaa_assert(    condicion4, \
                            mensajeFalse = "Características de test: no son correctas.", \
                            mensajeTrue = "Características de test: resultado validado." )

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
            
# condicion = False
# mensaje = "Este ese el texto a mostrar en caso de condición falsa."
# validar_resultado( "test", condicion, mensaje )
