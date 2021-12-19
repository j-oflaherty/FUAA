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

_THRESHOLD = 10e-9

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
def visualizar_costo_entrenamiento(costo_entrenamiento, xlabel='Número de iteración', 
                      ylabel='Costo de entrenamiento', title='Costo de entrenamiento'):
    '''
    Entrada:
        costo_entrenamiento: vector de dimención Niter con el costo en las Niter
                             iteraciones de entrenamiento de un modelo. 
    '''
    plt.figure(figsize=(8,8))
    plt.plot(costo_entrenamiento, '*-' )
    plt.xlabel( xlabel )
    plt.ylabel( ylabel )
    plt.title( title )
    plt.grid()


###############################################################################
def visualizar_frontera_decision(X, y, w, transformacion):
    '''
    Entrada:
        X: matriz de Nx3 que contiene los puntos en el espacio original
        y: etiquetas de los puntos
        w: vector de tamaño 10 que contiene los parámetros encontrados
    '''

    # Se construye una grilla de 50x50 en el dominio de los datos
    xs = np.linspace( X[:,1].min(), X[:,1].max())
    ys = np.linspace( X[:,2].min(), X[:,2].max())

    XX, YY = np.meshgrid( xs, ys ) 
    Z = np.zeros_like(XX)
    
    # se transforman los puntos de la grilla
    pts_grilla = np.vstack( (np.ones(XX.size), XX.ravel(),YY.ravel()) ).T
    pts_grilla_transformados = transformacion( pts_grilla )
    
    # los puntos transformados son proyectados utilizando el w
    Z = pts_grilla_transformados @ w
    Z = Z.reshape(XX.shape)#
    
    # se grafica la frontera de decisión, es decir, la línea de nivel 0  
    plt.figure(figsize=(8,8))
    plt.axis('equal')
    plt.contour(XX, YY, Z, [0])
    plt.scatter(X[:,1][y==1],X[:,2][y==1], s=40, color='b', marker='o', 
                label='etiqueta -1')
    plt.scatter(X[:,1][y==-1],X[:,2][y==-1], s=40, color='r', marker='x', 
                label='etiqueta 1')
    plt.title( 'Frontera de decision obtenida mediante\n%s()' % (transformacion.__name__) )
    

###############################################################################
def generar_semianillos(N, radio1, radio2, ancho1, ancho2, delta_x, delta_y):
    '''
    Entrada:
        N: número de muestras a generar
        radio1: radio interior del semicírculo asociado a la clase 1
        radio2: radio interior del semicírculo asociado a la clase 2
        ancho1: diferencia entre el radio exterior e interior del semicírculo asociado a la clase 1
        ancho2: diferencia entre el radio exterior e interior del semicírculo asociado a la clase 2
        delta_x: corrimiento en x respecto al origen del semicirculo asociado a la clase 2
        delta_y: corrimiento en y respecto al origen del semicirculo asociado a la clase 2

    Salida:
        X: matriz de Nx3 que contiene los datos generados en coordenadas homogéneas
        y: etiquetas asociadas a los datos
    '''
        
    X = np.ones((N, 3))
    # se sortea a que clase pertenecen las muestras
    y = 2 * (np.random.rand(N) < 0.5) - 1
    
    # radios y ángulos del semicírculo superior
    radios = radio1*(y==1) + radio2*(y==-1) + (ancho1*(y==1)+ancho2*(y==-1)) * np.random.rand(N)
    thetas = np.pi * np.random.rand(N)
    # coordenadas en x de ambos semicírculos
    X[:,1] = radios * np.cos(thetas) * y + (radio2 + ancho2/2 + delta_x)*(y==-1)
    # coordenadas en y de ambos semicírculos
    X[:,2] = radios * np.sin(thetas) * y - delta_y * (y==-1)
    
    return X, y


##################################################################################
def validar_resultado(*args, **kwargs):
    """"
    Función para validar resultado a invocar desde el notebook.
    """
    # _ES_CORRECTO = False
    _DEBUG = False

    # Abrir el cartel del validación.
    print( "+-------------------------------------------------------------------------+" )
    print( "|               FuAA (1er. parcial 2021): validar resultado               |" )
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
    # Ejercicio 1. Error binario.
    ###########################################################
    elif (args[0] == "error_binario"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
        else:
            print( "[validar_resultado] Error: llamada sin argumentos (%s)" % args[0] )
            return

        y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
        y_pred = np.array([-1, 1, 1, 1, 1, -1, -1, -1, -1, 1])
        err = test_algoritmo(y, y_pred)

        # Validar dimensión de la salida.
        condicion1 = not isinstance(err, np.ndarray)
        fuaa_assert(   condicion1, \
                    mensajeFalse = "La salida no tiene la dimensión correcta.", \
                    mensajeTrue= "Dimensión de salida: resultado validado.")

        # Validar transformación.
        condicion2 = err == 0.2
        fuaa_assert(   condicion2, \
                    mensajeFalse = "Error binario: resultado no validado.", 
                    mensajeTrue = "Error binario: resultado validado.")

    ###########################################################
    # Ejercicio 1. Transformación no lineal.
    ###########################################################
    elif (args[0] == "transformar_usando_polinomio"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
        else:
            print( "[validar_resultado] Error: llamada sin argumentos (%s)" % args[0] )
            return

        pts = np.array([[1, 1, 1], [1, 2, 3], [1, 0, 2], [1, 2, 0] ])
        pts_t_true = np.array( [[1., 1., 1., 1., 1., 1.], 
                                [1., 2., 3., 4., 6., 9.],
                                [1., 0., 2., 0., 0., 4.],
                                [1., 2., 0., 4., 0., 0.]] )
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
    # Ejercicio 1. Costos final nulo.
    ###########################################################
    elif (args[0] == "costo_final_nulo"):
        if validar_parametros( args, 2 ):
            costo = args[1]

            fuaa_assert( costo[-1] == 0, \
                mensajeFalse = "El costo final en datos linealmente separables debe ser 0." ,
                mensajeTrue = "Resultado validado: costo final nulo con datos linealmente separables.")

    ###########################################################
    # Ejercicio 1. Partición de conjuntos de train y validation.
    ###########################################################
    elif (args[0] == "train_val_sets"):
        if validar_parametros( args, 5 ):
            X_no_lin = args[1]
            Xtrain = args[2]
            Xvalid = args[3]
            valid_size = args[4]

            N0 = X_no_lin.shape[0]
            N1 = Xtrain.shape[0]
            N2 = Xvalid.shape[0]

            condicion = son_iguales( N1+N2, N0 ) and son_iguales( N2/N0, valid_size ) 
            fuaa_assert ( condicion,
                          mensajeFalse = "Los tamaños de las particiones no son correctos.",
                          mensajeTrue = "Resultado validado: particiones de tamaños correctos.")

    ###########################################################
    # Ejercicio 1. Transformación a polares.
    ###########################################################
    elif (args[0] == "trasformar_a_coordenadas_polares"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
        else:
            print( "[validar_resultado] Error: llamada sin argumentos (%s)" % args[0] )
            return

        pts = np.array([[1, 1, 1], [1, 2, 3], [1, 0, 2], [1, 2, 0] ])
        pts_t_true = np.array( [[1., 1.41421356, 0.78539816],
                                [1., 3.60555128, 0.98279372],
                                [1., 2., 1.57079633],
                                [1., 2., 0.]] )

        pts_t = test_algoritmo( pts )

        # Validar dimensión de datos transformados.
        condicion1 = pts_t.shape == pts_t_true.shape
        fuaa_assert(   condicion1, 
                    mensajeFalse = "La dimensión de los datos transformados no es correcta.", \
                    mensajeTrue= "Dimensión de salida: resultado validado.")

        # Validar transformación.
        condicion2 = son_iguales( pts_t_true, pts_t )
        fuaa_assert(   condicion2, 
                    mensajeFalse = "La transformación implementada no es correcta.", 
                    mensajeTrue = "Transformación: resultado validado.")


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
            
