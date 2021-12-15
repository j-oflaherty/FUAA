#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 07:52:52 2021

@author: fuaa
"""
import numpy as np
import h5py

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
        printcolor( "[validar_resultado] Insuficientes parámetros (\"%s\"), se necesitan %d, hay %d." % \
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
    # Práctico 4. Transformación no lineal.
    ###########################################################
    elif (args[0] == "transformada_no_lineal"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
        else:
            print( "[validar_resultado] Error: llamada sin argumentos (%s)" % args[0] )
            return

        pts = np.array([[1, 1, 1], [1, 2, 3], [1, 0, 2], [1, 2, 0] ])
        pts_t_true = np.array( [[1., 1., 1., 1., 1., 1., 0., 2.],
                                [1., 2., 3., 4., 9., 6., 1., 5.],
                                [1., 0., 2., 0., 4., 0., 2., 2.],
                                [1., 2., 0., 4., 0., 0., 2., 2.]] )
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
    # Práctico 4. Transformación polinómica.
    ###########################################################
    elif (args[0] == "transformacion_polinomica"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
        else:
            print( "[validar_resultado] Error: llamada sin argumentos (%s)" % args[0] )
            return

        pts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        pts_t_true = np.array( [[ 1.,  1.,  1.],[ 1.,  2.,  4.],[ 1.,  3.,  9.],
                                [ 1.,  4., 16.],[ 1.,  5., 25.],[ 1.,  6., 36.],
                                [ 1.,  7., 49.],[ 1.,  8., 64.],[ 1.,  9., 81.]] )
        
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
    # Práctico 4. Transformación polinómica 2.
    ###########################################################
    elif (args[0] == "Xpoly2"):
        if validar_parametros( args, 3 ):
            Xpoly = args[1]
            w_ls = args[2]
            fuaa_assert( Xpoly.shape[1] == 3, \
                mensajeFalse = "La dimensión del vector transformado debe ser 3 para un polinomio de grado 2." )

            fuaa_assert( len( w_ls ) == 3, \
                mensajeFalse = "La dimensión del vector de pesos debe ser 3 para un polinomio de grado 2." )

    ###########################################################
    # Práctico 4. Estandarizar características
    ###########################################################
    elif (args[0] == "estandarizar_caracteristicas"):
        if 'funcion' in kwargs:
            estandarizar_caracteristicas = kwargs['funcion']

        caract_uniformes = np.random.rand(100,5)
        caract_uniformes[:,0] = 1
        caract_estandarizadas, mu_sigma = estandarizar_caracteristicas(caract_uniformes)

        # Se verifica el correcto funcionamiento cuando no se le pasan valores de mu_sigma
        fuaa_assert(np.allclose( caract_estandarizadas[:,1:].mean(axis=0), 0 ),  
                    mensajeFalse = 'Las características deben tener media cero.', 
                    mensajeTrue = 'Las características tienen media cero.')

        fuaa_assert(np.allclose( caract_estandarizadas[:,1:].std(axis=0), 1 ),
                    mensajeFalse = 'Las características deben tener desviación estandar 1.',
                    mensajeTrue = 'Las características tienen desviación estandar 1.')

        fuaa_assert(np.allclose( caract_estandarizadas[:,0], 1 ),
                    mensajeFalse = 'Las características deben expresarse en coordenadas homogéneas.',
                    mensajeTrue = 'Las características expresadas en coordenadas homogéneas.')

        caract_uniformes_test = np.random.rand( 100, 5 )
        caract_uniformes_test[:,0] = 1
        caract_estandarizadas_test, mu_sigma_test = estandarizar_caracteristicas( caract_uniformes_test, mu_sigma )

        # se verifica el correcto funcionamiento cuando se le pasan valores de mu_sigma
        fuaa_assert(np.allclose( mu_sigma, mu_sigma_test ),
                    mensajeFalse = 'Cuando se pasan valores de mu_sigma se deben devolver los mismos valores de mu_sigma.',
                    mensajeTrue = 'Retorno de mu_sigma correcto.')
        
        caract_uniformes_test_std = (caract_uniformes_test[:,1:] - mu_sigma[0]) / mu_sigma[1]
        fuaa_assert(np.allclose( caract_uniformes_test_std, caract_estandarizadas_test[:,1:] ),
                        mensajeFalse = 'La estandarización no es correcta cuando se pasan valores de mu_sigma.',
                        mensajeTrue = 'Estandarización validada con pasaje de mu_sigma.')

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
