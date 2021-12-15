#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 07:52:52 2021

@author: fuaa
"""
import numpy as np
from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation

# import h5py
# from algoritmos import generar_semianillos_validacion

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

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Validar número de parámetros.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def validar_parametros( parametros, min_params ):
    condicion = (len( parametros ) >= min_params)
    if not condicion:
        printcolor( "[validar_resultado] Insuficientes parámetros (\"%s\"), se necesitan %d,  hay %d." % \
            (parametros[0], min_params, len(parametros)), "r" )
    return condicion

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Función para validar resultado a invocar desde el notebook.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
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

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Visualización de la función a optimizar en rango de interés.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def visualizar_funcional(grid_step = 0.1, rango = 2):
    xs = np.arange(-rango, rango, grid_step)
    yes = np.arange(-rango, rango, grid_step)
    xx, yy = np.meshgrid(xs, yes)
    z = xx**2 + 2*yy**2 + 2*np.sin(2*np.pi*xx) * 2*np.sin(2*np.pi*yy)

    fig = plt.figure( figsize = (5,5) )
    ax = fig.add_subplot( 111, projection='3d' )
    ax.plot_surface( xx, yy, z )
    plt.xlabel( 'x' )
    plt.ylabel( 'y' )
    plt.title( 'Funcional con grid_step = %.2f' % grid_step )
    plt.tight_layout()
    
    return True