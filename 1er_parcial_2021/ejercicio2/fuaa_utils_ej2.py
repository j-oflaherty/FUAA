#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 07:52:52 2021

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
    if x1.shape == x2.shape:
        if isinstance(x1, np.ndarray):
            dif = np.sqrt(np.sum( ( x1 - x2 )**2 )) / x1.size
        elif isinstance(x1, float):
            dif = np.abs( x1 - x2 )
        condicion = (dif < threshold)
    else:
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
def visualizar_experimentos_modelo_lineal(  N_max, K, reg_factor, lista_sesgos, lista_varianzas, lista_Ein, 
                                            lista_Ein_modelo_regularizado, lista_sesgos_modelo_regularizado, 
                                            lista_varianzas_modelo_regularizado, lista_Eout, 
                                            lista_Eout_modelo_regularizado, mostrar2 = True ):
        
    
    if mostrar2:
        plt.figure(figsize=(15,5))
        plt.subplot(1,2,1)
        plt.plot(np.arange(2, N_max), lista_sesgos, '*-', label='sesgo')
        plt.fill_between(np.arange(2,N_max), lista_sesgos, lista_Eout, \
                        facecolors='#b2dfee', label='varianza')
        plt.plot(np.arange(2, N_max), lista_Ein, '*-', label='Ein')
        plt.plot(np.arange(2, N_max), lista_Eout,'*-', label='Eout')
        plt.xlabel('Número de puntos')
        plt.title('Modelo sin regularizar')
        plt.legend()
        plt.grid()
        plt.ylim(0,2)
        
        plt.subplot(1,2,2)
    else:
        plt.figure(figsize=(7,5))

    plt.plot(np.arange(2, N_max), lista_sesgos_modelo_regularizado, '*-', label='sesgo_reg')
    plt.fill_between(np.arange(2,N_max), lista_sesgos_modelo_regularizado, \
                     lista_Eout_modelo_regularizado, facecolors='#b2dfee', label='varianza')
    plt.plot(np.arange(2, N_max), lista_Ein_modelo_regularizado, '*-', label='Ein')
    plt.plot(np.arange(2, N_max), lista_Eout_modelo_regularizado,'*-', label='Eout')
    plt.xlabel('Número de puntos')
    if (reg_factor == 0):
        title_str = 'Modelo sin regularizar ($\lambda=%d$)'
    else:
        title_str = 'Modelo regularizado ($\lambda=%.2e$)'
    plt.title( title_str % reg_factor)
    plt.legend()
    plt.grid()
    plt.ylim(0,2)
    plt.show()


##################################################################################
def mostrar_experimento(a, b, nombre_experimento):
    '''
    Se grafica los resultados del experimento junto con la función.
    '''

    x = np.linspace(-1, 1)
    N = len(a)
    plt.figure(figsize=(7,5))
    for n in range(0, N, N // 100):
        plt.plot(x, a[n] * x + b[n], 'y', alpha=0.2)

    plt.plot(x, np.sin(np.pi * x), 'b')
    plt.ylim(-1, 1)
    plt.title(nombre_experimento)


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
    # Ejercicio 2. Regresión Lineal
    ###########################################################
    elif (args[0] == "regresion_lineal"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
        else:
            print( "[validar_resultado] Error: llamada sin argumentos (%s)" % args[0] )
            return
        
        np.random.seed(43)
        y3 = np.random.randn(9)
        Z3 = np.random.randn(9,7)
        Z3[:,0]=1
        w3 = test_algoritmo(Z3, y3 )
        w3_true = np.array([ 0.99480683, -1.02402373,  0.40146952,  0.21686739,  0.23815722, -1.23370181,
 -0.64320162])
        condicion1 = son_iguales( w3, w3_true )
        fuaa_assert(   condicion1, \
                    mensajeFalse = "Revisar la implementación de la regresión lineal", \
                    mensajeTrue = "Implementación de la regresión lineal: resultado validado.")

    ###########################################################
    # Ejercicio 2. Regresión Ridge
    ###########################################################
    elif (args[0] == "regresion_Ridge"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
        else:
            print( "[validar_resultado] Error: llamada sin argumentos (%s)" % args[0] )
            return

        np.random.seed(43)
        y3 = np.random.randn(9)
        Z3 = np.random.randn(9,7)
        Z3[:,0] = 1
        reg = 3.3
        w3 = test_algoritmo(Z3, y3, reg )
        w3_true = np.array([ 0.58610027, -0.09425186,  0.12976429,  0.11544501,  0.14085992, -0.39307442,
 -0.20333349])
        condicion1 = son_iguales( w3, w3_true )
        fuaa_assert(   condicion1, \
                    mensajeFalse = "Revisar la implementación de la regresión lineal regularizada", \
                    mensajeTrue= "Implementación de la regresión lineal regularizada: resultado validado.")

    ###########################################################
    # Ejercicio 2. Obtener hipótesis.
    ###########################################################
    elif (args[0] == "obtener_hipotesis"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
        else:
            print( "[validar_resultado] Error: llamada sin argumentos (%s)" % args[0] )
            return

        np.random.seed(42)
        K = 10
        x1 = np.random.randn(K)
        y1 = np.random.randn(K)
        x2 = np.random.randn(K)
        y2 = np.random.randn(K)
        a, b = test_algoritmo(x1,y1,x2,y2,'H0')
        condicion_a = son_iguales( a, np.zeros(K) )
        condicion_b = son_iguales( b, np.array([-0.53256215, 0.69327422, 0.11423252, -1.48549559, -0.45118646, -0.89156559, -0.40198376, -0.8227114, -1.11810506, -0.60772123]) )
        fuaa_assert( condicion_a, 
                    mensajeFalse = "Parámetros 'a' con H0 no validados.", 
                    mensajeTrue = "Parámetros 'a' con H0 validados." )
        fuaa_assert( condicion_b,
                    mensajeFalse = "Parámetros 'b' con H0 no validados.", 
                    mensajeTrue = "Parámetros 'b' con H0 validados." )

        a, b = test_algoritmo(x1,y1,x2,y2,'H1')
        condicion_a = son_iguales( a, np.array([-0.14272265, -26.48788688, 0.44032569, -0.29024211, -8.21154654, -1.90852891, -0.44747339, 5.80470861, 3.2033256, -1.92886739]) )
        condicion_b = son_iguales( b, np.array([-0.39252533, -4.12805892, -0.04323163, -1.47123285, -3.64767917, -1.00914468, -0.30617541, -4.14048764, 0.59585525, -0.36577733]) )
        fuaa_assert( condicion_a,
                    mensajeFalse = "Parámetros 'a' con H1 no validados.", 
                    mensajeTrue = "Parámetros 'a' con H1 validados." )
        fuaa_assert( condicion_b,
                    mensajeFalse = "Parámetros 'b' con H1 no validados.", 
                    mensajeTrue = "Parámetros 'b' con H1 validados." )

    ###########################################################
    # Ejercicio 2. MSE.
    ###########################################################
    elif (args[0] == "mse"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
        else:
            print( "[validar_resultado] Error: llamada sin argumentos (%s)" % args[0] )
            return

        np.random.seed(43)
        N = 10
        d = 4
        X = np.random.rand(N,d+1)
        w = np.random.rand(d+1)
        y = np.random.rand(N)
        err = test_algoritmo(X,w,y)
        condicion = (err == 1.2015737362409753)
        fuaa_assert( condicion,
                    mensajeFalse = "El MSE calculado no es correcto.",
                    mensajeTrue = "Implementación de mse(): resultado validado." )


    ###########################################################
    # Ejercicio 2. Calculo de sesgo y varianza
    ###########################################################
    elif (args[0] == "calcular_sesgo_varianza"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
        else:
            print( "[validar_resultado] Error: llamada sin argumentos (%s)" % args[0] )
            return

        np.random.seed(43)
        Ntest = 10
        K = 14
        X = np.random.rand( Ntest )
        a = np.random.rand( K )
        b = np.random.rand( K )
        sesgo, varianza = test_algoritmo( X, a, b )

        condicion_sesgo = ( sesgo == 0.07425087677513127 )
        fuaa_assert( condicion_sesgo,
                    mensajeFalse = "El 'sesgo' calculado no es correcto.",
                    mensajeTrue = "Implementación de cálculo del sesgo: resultado validado." )

        condicion_varianza = ( varianza == 0.14012245581195754 )
        fuaa_assert( condicion_varianza,
                    mensajeFalse = "La 'varianza' calculado no es correcto.",
                    mensajeTrue = "Implementación de cálculo de la varianza: resultado validado." )

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
