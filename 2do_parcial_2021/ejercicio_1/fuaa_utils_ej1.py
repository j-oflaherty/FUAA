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

#########################################################################################

# funciones auxiliares (Ejecutar y seguir)
def error_relativo(x, y):
    ''' devuelve el error relativo'''
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

#########################################################################################

def calcular_gradiente_numerico_array(f, x, df, h=1e-5):
    '''
    Evalúa el gradiente numérico para una función que acepta un arreglo numpy y
    devuelve un arreglo numpy.
    '''
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad

##########################################################################################

def calcular_gradiente_numerico(f, x, verbose=True, h=0.00001):
    '''
    Evalúa el gradiente numérico de f en x
    - f es una función que recibe un solo argumente
    - x es el punto (numpy array) en que se evalúa el gradiente
    '''
    
    # se inicializa el gradiente 
    grad = np.zeros_like(x)
    # se define un iterador sobre todos los elementos de x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # se evalúa la función en x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h # se suma h al valor original de x
        fxph = f(x) # se evalúa f(x + h)
        x[ix] = oldval - h
        fxmh = f(x) # se evalúa f(x - h)
        x[ix] = oldval # se restaura el valor original de x

        # se calcula la derivada parcial con la fórmula centrada
        grad[ix] = (fxph - fxmh) / (2 * h) 
        if verbose:
            print(ix, grad[ix])
        it.iternext() # step to next dimension

    return grad

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

def mostrar_ajuste(y, y_pred, ax, title):
    '''
    Entrada:
        y: vector que contiene los valores objetivo
        y_pred: vector que contiene los valores estimados
    
    '''
    ax.scatter(y, y_pred, s=2, label='predicciones')
    ax.set_xlabel("Valores reales", fontsize=18)
    ax.set_ylabel("Valores estimados", fontsize=18)
    ax.plot(np.linspace(y.min(),y.max()),np.linspace(y.min(),y.max()), c='r', label='estimaciones ideales')
    ax.set_title(title, fontsize=18)
    ax.legend()
    ax.grid()
    ax.set_ylim(2,10)

###################################################################################

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


    #####################################################################
    # Práctico 5. Inicialización de pesos.
    #####################################################################
    elif (args[0] == "inicializar_pesos"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
        else:
            print("[validar_resultado] Error: llamada sin argumentos (%s)" % args[0])
            return

        # Se testea la inicialización con pesos aleatorios
        d0, d1, d2, d3 = 3, 6, 3, 1
        W1_correcto = np.array([[ 0.93781623, -0.35319773, -0.3049401 , -0.61947872,  0.49964333, -1.32879399],
                       [ 1.00736754, -0.43948301,  0.18419731, -0.14397405,  0.84414841, -1.18942279],
                         [-0.18614766, -0.22173389,  0.65458209, -0.63502252, -0.09955147, -0.50683179]])
        b1_correcto = np.array([0., 0., 0., 0., 0., 0.])
        W2_correcto = np.array([[ 0.01723369,  0.23793331, -0.4493259 ],[ 0.4673315 ,  0.36807287,  0.20514245],
                       [ 0.3677729 , -0.27913073, -0.05016972], [-0.38202627, -0.10936485,  0.21651671],
                       [-0.28236932, -0.16197395, -0.28053708], [-0.34505376, -0.27403509, -0.0051703 ]])
        b2_correcto = np.array([0., 0., 0.])  
        W3_correcto = np.array([[-0.64507943],[ 0.13533997],[ 0.95828723]])
        b3_correcto = np.array([0.])

        parametros = test_algoritmo((d0, d1, d2, d3))
        W1 = parametros['W1']
        b1 = parametros['b1']
        W2 = parametros['W2']
        b2 = parametros['b2']
        W3 = parametros['W3']
        b3 = parametros['b3']


        # Chequear dimensiones.
        fuaa_assert(W1.shape == (d0, d1), mensajeFalse="Dimensiones de W1 no validadas.", mensajeTrue="Dimensiones de W1 validadas.")
        fuaa_assert(b1.shape == (d1, ), mensajeFalse="Dimensiones de b1 no validadas.", mensajeTrue="Dimensiones de b1 validadas.")
        fuaa_assert(W2.shape == (d1, d2), mensajeFalse="Dimensiones de W2 no validadas.", mensajeTrue="Dimensiones de W2 validadas.")
        fuaa_assert(b2.shape == (d2, ), mensajeFalse="Dimensiones de b2 no validadas.", mensajeTrue="Dimensiones de b2 validadas.")
        fuaa_assert(W3.shape == (d2, d3), mensajeFalse="Dimensiones de W3 no validadas.", mensajeTrue="Dimensiones de W3 validadas.")
        fuaa_assert(b3.shape == (d3, ), mensajeFalse="Dimensiones de b3 no validadas.", mensajeTrue="Dimensiones de b3 validadas.")

        # Chequear generación de los pesos.
        fuaa_assert(son_iguales(W1, W1_correcto), mensajeFalse="Cálculo de W1 no validado.", mensajeTrue="Cálculo de W1 validado.")
        fuaa_assert(son_iguales(b1, b1_correcto), mensajeFalse="Cálculo de b1 no validado.", mensajeTrue="Cálculo de b1 validado.")
        fuaa_assert(son_iguales(W2, W2_correcto), mensajeFalse="Cálculo de W2 no validado.", mensajeTrue="Cálculo de W2 validado.")
        fuaa_assert(son_iguales(b2, b2_correcto), mensajeFalse="Cálculo de b2 no validado.", mensajeTrue="Cálculo de b2 validado.")
        fuaa_assert(son_iguales(W3, W3_correcto), mensajeFalse="Cálculo de W3 no validado.", mensajeTrue="Cálculo de W3 validado.")
        fuaa_assert(son_iguales(b3, b3_correcto), mensajeFalse="Cálculo de b3 no validado.", mensajeTrue="Cálculo de b3 validado.")

    #####################################################################
    # Práctico 5. Validación afín forward.
    #####################################################################
    elif (args[0] == "afin_forward"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
        else:
            print("[validar_resultado] Error: llamada sin argumentos (%s)" % args[0])
            return

        num_muestras = 2
        dim_entrada = 120
        dim_salida = 3

        X = np.linspace(-0.1, 0.5, num=(num_muestras * dim_entrada)).reshape(num_muestras, dim_entrada)
        W = np.linspace(-0.2, 0.3, num=(dim_entrada * dim_salida)).reshape(dim_entrada, dim_salida)
        b = np.linspace(-0.3, 0.1, num=dim_salida)

        S, _ = test_algoritmo(X, W, b)
        S_correcto = np.array([[ 1.49834967,  1.70660132,  1.91485297],
                                [ 3.25553199,  3.5141327,   3.77273342]])

        # Validar dimensiones.
        fuaa_assert(S.shape == (X.shape[0], W.shape[1]), 
                    mensajeFalse="Dimensión de la salida no validada.", 
                    mensajeTrue="Dimensión de la salida validada.")

        # Validar resultado de salida.
        fuaa_assert(son_iguales(S, S_correcto,1e-8),
                    mensajeFalse="Cálculo de S no validada.", 
                    mensajeTrue="Resultado validado.")


    #####################################################################
    # Práctico 5. Validación de la sigmoide.
    #####################################################################
    elif (args[0] == "sigmoide"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
        else:
            print("[validar_resultado] Error: llamada sin argumentos (%s)" % args[0])
            return

        S = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)
        X, S_ = test_algoritmo(S)
        X_correcto = np.array([[0.37754067, 0.39913012, 0.42111892, 0.44342513],
                               [0.46596182, 0.48863832, 0.51136168, 0.53403818],
                               [0.55657487, 0.57888108, 0.60086988, 0.62245933]])

        # Validar dimensiones.
        fuaa_assert(X.shape == S.shape, mensajeFalse="Las dimensiones de la entrada y la salida deben ser igulaes.", 
                                        mensajeTrue="Dimensiones de la salida validadas.")

        # Validar cache.
        fuaa_assert(son_iguales(S,S_), mensajeFalse="La salida \"cache\" debe ser igual a la entrada.", 
                                       mensajeTrue="Salida \"cache\" validada.")

        # Se compara la salida con la nuestra. El error debería ser del orden de 10^{-8}
        fuaa_assert(error_relativo(X, X_correcto) < 10e-7, mensajeFalse="Cálculo de la salida no validado.", 
                                                           mensajeTrue="Cálculo de la salida validado.")


    #####################################################################
    # Práctico 5. Validación de la tangente hiperbólica.
    #####################################################################
    elif (args[0] == "tanh"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
        else:
            print("[validar_resultado] Error: llamada sin argumentos (%s)" % args[0])
            return

        S = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)
        X, S_ = test_algoritmo(S)
        X_correcto = np.array([ [-0.46211716, -0.38770051, -0.30786199, -0.22343882],
                                [-0.13552465, -0.04542327,  0.04542327,  0.13552465],
                                [ 0.22343882,  0.30786199,  0.38770051,  0.46211716]])

        # Validar dimensiones.
        fuaa_assert(X.shape == S.shape, mensajeFalse="Las dimensiones de la entrada y la salida deben ser igulaes.", 
                                        mensajeTrue="Dimensiones de la salida validadas.")

        # Validar cache.
        fuaa_assert(son_iguales(S,S_), mensajeFalse="La salida \"cache\" debe ser igual a la entrada.", 
                                       mensajeTrue="Salida \"cache\" validada.")

        # Se compara la salida con la nuestra. El error debería ser del orden de 10^{-8}
        fuaa_assert(error_relativo(X, X_correcto) < 10e-7, mensajeFalse="Cálculo de la salida no validado.", 
                                                           mensajeTrue="Cálculo de la salida validado.")


    #####################################################################
    # Práctico 5. Validación de la Rectified Linear Unit.
    #####################################################################
    elif (args[0] == "relu"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
        else:
            print("[validar_resultado] Error: llamada sin argumentos (%s)" % args[0])
            return

        S = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)
        X, S_ = test_algoritmo(S)
        X_correcto = np.array( [[ 0.,          0.,          0.,          0.,        ],
                                [ 0.,          0.,          0.04545455,  0.13636364,],
                                [ 0.22727273,  0.31818182,  0.40909091,  0.5,       ]])

        # Validar dimensiones.
        fuaa_assert(X.shape == S.shape, mensajeFalse="Las dimensiones de la entrada y la salida deben ser igulaes.", 
                                        mensajeTrue="Dimensiones de la salida validadas.")

        # Validar cache.
        fuaa_assert(son_iguales(S,S_), mensajeFalse="La salida \"cache\" debe ser igual a la entrada.", 
                                       mensajeTrue="Salida \"cache\" validada.")

        # Se compara la salida con la nuestra. El error debería ser del orden de 10^{-8}
        fuaa_assert(error_relativo(X, X_correcto) < 10e-7, mensajeFalse="Cálculo de la salida no validado.", 
                                                           mensajeTrue="Cálculo de la salida validado.")


    #####################################################################
    # Práctico 5. Validación de la capa Afin->Activación.
    #####################################################################
    elif (args[0] == "afin_activacion_forward"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
        else:
            print("[validar_resultado] Error: llamada sin argumentos (%s)" % args[0])
            return

        num_muestras = 3
        dim_entrada = 80
        dim_salida = 2
        X_prev = np.linspace(-0.1, 0.5, num=(num_muestras * dim_entrada)).reshape(num_muestras, dim_entrada)
        W = np.linspace(-0.2, 0.3, num=(dim_entrada * dim_salida) ).reshape(dim_entrada, dim_salida)
        b = np.linspace(-0.3, 0.1, num=dim_salida)

        # Sigmoide.
        X, _ = test_algoritmo(X_prev, W, b, 'sigmoide')
        X_correcto = np.array([[0.59153611, 0.6835444 ],
                               [0.75921928, 0.8318392 ],
                               [0.87286027, 0.91888762]])
        fuaa_assert(X.shape == (X_prev.shape[0], W.shape[1]), mensajeFalse="Dimensiones de la salida sigmoide no validada.", 
                                                              mensajeTrue="Dimensiones de la salida sigmoide validada.")
        fuaa_assert(error_relativo(X, X_correcto) < 10e-8, mensajeFalse="Cálculo de la salida sigmoide no validado.", 
                                                           mensajeTrue="Cálculo de la salida sigmoide validado.")

        # Tangente hiperbólica.
        X, _ = test_algoritmo(X_prev, W, b, 'tanh')
        X_correcto = np.array([[0.35427088, 0.64699264],
                               [0.81722466, 0.92147542],
                               [0.95844863, 0.98453647],])
        fuaa_assert(X.shape == (X_prev.shape[0], W.shape[1]), mensajeFalse="Dimensiones de la salida tanh no validada.", 
                                                              mensajeTrue="Dimensiones de la salida tanh validada.")
        fuaa_assert(error_relativo(X, X_correcto) < 10e-8, mensajeFalse="Cálculo de la salida tanh no validado.", 
                                                           mensajeTrue="Cálculo de la salida tanh validado.")

        # Rectified Linear Unit.
        X, _ = test_algoritmo(X_prev, W, b, 'relu')
        X_correcto = np.array([[[0.3703192,  0.77010868],
                                [1.14840399, 1.59871845],
                                [1.92648878, 2.42732823]],])
        fuaa_assert(X.shape == (X_prev.shape[0], W.shape[1]), mensajeFalse="Dimensiones de la salida relu no validada.", 
                                                              mensajeTrue="Dimensiones de la salida relu validada.")
        fuaa_assert(error_relativo(X, X_correcto) < 10e-8, mensajeFalse="Cálculo de la salida relu no validado.", 
                                                           mensajeTrue="Cálculo de la salida relu validado.")


    #####################################################################
    # Práctico 5. Validación del error cuadratico medio como función de 
    # costo de una red neuronal.
    #####################################################################
    elif (args[0] == "mse"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
        else:
            print("[validar_resultado] Error: llamada sin argumentos (%s)" % args[0])
            return

        np.random.seed(42)
        num_inputs = 9
        xL = np.random.rand(num_inputs, 1)
        y = np.random.rand(num_inputs, 1)
        costo_true = 0.07937718108178748

        dxL_num = calcular_gradiente_numerico(lambda xL: test_algoritmo(xL, y)[0], xL, verbose=False)
        costo, dxL = test_algoritmo(xL, y)

        fuaa_assert(costo.shape == (), mensajeFalse="Dimensiones de costo no validadas.", 
                                       mensajeTrue="Dimensiones de costo validadas.")
        fuaa_assert(dxL.shape == xL.shape, mensajeFalse="Dimensiones de dxL no validadas.",
                                           mensajeTrue="Dimensiones de dxL validadas.")
        fuaa_assert(son_iguales(costo, costo_true), mensajeFalse="Cálculo del costo no validado.", 
                                                    mensajeTrue="Cálculo del costo validado.")
        fuaa_assert(error_relativo(dxL_num, dxL) < 1e-9, mensajeFalse="Cálculo del gradiente no validado.", 
                                                         mensajeTrue="Cálculo del gradiente validado.")


    #####################################################################
    # Práctico 5. Validación del cálculo de la entropía cruzada.
    #####################################################################
    elif (args[0] == "entropia_cruzada"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
        else:
            print("[validar_resultado] Error: llamada sin argumentos (%s)" % args[0])
            return

        # Se testea la implementación de la entropía cruzada
        np.random.seed(231)
        num_classes, num_inputs = 2, 10
        xL = np.random.rand(num_inputs, 1)
        y = np.random.randint(num_classes, size=(num_inputs, 1))
        costo_true = 1.05681162393179

        dxL_num = calcular_gradiente_numerico(lambda xL: test_algoritmo(xL, y)[0], xL, verbose=False)
        costo, dxL = test_algoritmo(xL, y)

        fuaa_assert(costo.shape == (), mensajeFalse="Dimensiones de costo no validadas.", 
                                       mensajeTrue="Dimensiones de costo validadas.")
        fuaa_assert(dxL.shape == xL.shape, mensajeFalse="Dimensiones de dxL no validadas.",
                                           mensajeTrue="Dimensiones de dxL validadas.")
        fuaa_assert(son_iguales(costo, costo_true), mensajeFalse="Cálculo del costo no validado.", 
                                                    mensajeTrue="Cálculo del costo validado.")
        fuaa_assert(error_relativo(dxL_num, dxL) < 1e-7, mensajeFalse="Cálculo del gradiente no validado.", 
                                                         mensajeTrue="Cálculo del gradiente validado.")


    #####################################################################
    # Práctico 5. Validación de la capa afín backward.
    #####################################################################
    elif (args[0] == "afin_backward"):
        if 'funcion' in kwargs:
            test_algoritmo = kwargs['funcion']
            afin_forward = kwargs['f_forward']
        else:
            print("[validar_resultado] Error: llamada sin argumentos (%s)" % args[0])
            return

        np.random.seed(43)
        X = np.random.randn(10, 6)
        W = np.random.randn(6, 5)
        b = np.random.randn(5)
        dout = np.random.randn(10, 5)

        dX_num = calcular_gradiente_numerico_array(lambda X: afin_forward(X, W, b)[0], X, dout)
        dW_num = calcular_gradiente_numerico_array(lambda W: afin_forward(X, W, b)[0], W, dout)
        db_num = calcular_gradiente_numerico_array(lambda b: afin_forward(X, W, b)[0], b, dout)
        _, cache = afin_forward(X, W, b)
        dX, dW, db = test_algoritmo(dout, cache)

        fuaa_assert (dX.shape == X.shape, mensajeFalse="Dimensión del gradiente respecto a la activación de la capa anterior (dE/dX_prev) no validado.", 
                                          mensajeTrue="Dimensión del gradiente respecto a la activación de la capa anterior (dE/dX_prev) validado.")
        fuaa_assert (dW.shape == W.shape, mensajeFalse="Dimensión del gradiente respecto a W de la capa actual (dE/dW) no validado.", 
                                          mensajeTrue="Dimensión del gradiente respecto a W de la capa actual(dE/dW) validado.")
        fuaa_assert (db.shape == b.shape, mensajeFalse="Dimensión del gradiente respecto a \"b\" de la capa actual (dE/db) no validado.", 
                                          mensajeTrue="Dimensión del gradiente respecto a \"b\" de la capa actual (dE/db) validado.")
        fuaa_assert(error_relativo(dX_num, dX) < 1e-8, mensajeFalse="Gradiente respecto a la activación de la capa anterior (dE/dX_prev) no validado.", 
                                                       mensajeTrue="Gradiente respecto a la activación de la capa anterior (dE/dX_prev) validado.")
        fuaa_assert(error_relativo(dW_num, dW) < 1e-8, mensajeFalse="Gradiente respecto a W de la capa actual (dE/dW) no validado.", 
                                                       mensajeTrue="Gradiente respecto a W de la capa actual (dE/dW) validado.")
        fuaa_assert(error_relativo(db_num, db) < 1e-8, mensajeFalse="Gradiente respecto a \"b\" de la capa actual (dE/db) no validado.", 
                                                       mensajeTrue="Gradiente respecto a \"b\" de la capa actual (dE/db) validado.")


    #####################################################################
    # Práctico 5. Validación de las funciones de activación backward.
    #####################################################################

    elif (args[0] == "activacion_backward"):
        if 'f_backward' in kwargs:
            test_algoritmo = kwargs['f_backward']
            f_forward = kwargs['f_forward']
        else:
            print("[validar_resultado] Error: llamada sin argumentos (%s)" % args[0])
            return

        np.random.seed(231)
        S = np.random.randn(10, 10)
        dout = np.random.randn(*S.shape)
        dS_num = calcular_gradiente_numerico_array(lambda S: f_forward(S)[0], S, dout)

        _, cache = f_forward(S)
        dS = test_algoritmo(dout, cache)

        fuaa_assert(dS.shape == S.shape, mensajeFalse='Dimensiones del gradiente respecto a S no validadas.',
                                         mensajeTrue='Dimensiones del gradiente respecto a S validadas.')

        fuaa_assert(error_relativo(dS_num, dS) < 1e-9, mensajeFalse="Cálculo de la salida no validado.", 
                                                       mensajeTrue="Cálculo de la salida validado.")


    #####################################################################
    # Práctico 5. Validación de la capa Afin->Activación backwards.
    #####################################################################
    elif (args[0] == "afin_activacion_backward"):
        if 'f_backward' in kwargs:
            test_algoritmo = kwargs['f_backward']
            f_forward = kwargs['f_forward']
        else:
            print("[validar_resultado] Error: llamada sin argumentos (%s)" % args[0])
            return

        np.random.seed(231)
        x = np.random.randn(2, 12)
        w = np.random.randn(12, 10)
        b = np.random.randn(10)
        dout = np.random.randn(2, 10)
        activaciones = ['relu', 'sigmoide'] #['relu', 'sigmoide','tanh']

        for activacion in activaciones:
            out, cache = f_forward(x, w, b, activacion)
            dx, dw, db = test_algoritmo(dout, cache, activacion)

            dx_num = calcular_gradiente_numerico_array(lambda x: f_forward(x, w, b, activacion)[0], x, dout)
            dw_num = calcular_gradiente_numerico_array(lambda w: f_forward(x, w, b, activacion)[0], w, dout)
            db_num = calcular_gradiente_numerico_array(lambda b: f_forward(x, w, b, activacion)[0], b, dout)

            fuaa_assert(error_relativo(dx_num, dx), 
                        mensajeFalse="Cálculo del grandiente respecto a dX ("+activacion+") no validado.",
                        mensajeTrue="Cálculo del grandiente respecto a dX ("+activacion+") validado.")
            fuaa_assert(error_relativo(dw_num, dw), 
                        mensajeFalse="Cálculo del grandiente respecto a dW ("+activacion+") no validado.",
                        mensajeTrue="Cálculo del grandiente respecto a dW ("+activacion+") validado.")
            fuaa_assert(error_relativo(db_num, db), 
                        mensajeFalse="Cálculo del grandiente respecto a db ("+activacion+") no validado.",
                        mensajeTrue="Cálculo del grandiente respecto a db ("+activacion+") validado.")
            
    ###########################################################
    # 2do parcial 2021. Estandarizar características
    ###########################################################
    elif (args[0] == "estandarizar_caracteristicas"):
        if 'funcion' in kwargs:
            estandarizar_caracteristicas = kwargs['funcion']

        caract_uniformes = np.random.rand(100,5)
        caract_estandarizadas, mu_sigma = estandarizar_caracteristicas(caract_uniformes)

        # Se verifica el correcto funcionamiento cuando no se le pasan valores de mu_sigma
        fuaa_assert(np.allclose( caract_estandarizadas.mean(axis=0), 0 ),  
                    mensajeFalse = 'Las características deben tener media cero.', 
                    mensajeTrue = 'Las características tienen media cero.')

        fuaa_assert(np.allclose( caract_estandarizadas.std(axis=0), 1 ),
                    mensajeFalse = 'Las características deben tener desviación estandar 1.',
                    mensajeTrue = 'Las características tienen desviación estandar 1.')

        caract_uniformes_test = np.random.rand( 100, 5 )
        caract_estandarizadas_test, mu_sigma_test = estandarizar_caracteristicas( caract_uniformes_test, mu_sigma )

        # se verifica el correcto funcionamiento cuando se le pasan valores de mu_sigma
        fuaa_assert(np.allclose( mu_sigma, mu_sigma_test ),
                    mensajeFalse = 'Cuando se pasan valores de mu_sigma se deben devolver los mismos valores de mu_sigma.',
                    mensajeTrue = 'Retorno de mu_sigma correcto.')
        
        caract_uniformes_test_std = (caract_uniformes_test - mu_sigma[0]) / mu_sigma[1]
        fuaa_assert(np.allclose( caract_uniformes_test_std, caract_estandarizadas_test ),
                        mensajeFalse = 'La estandarización no es correcta cuando se pasan valores de mu_sigma.',
                        mensajeTrue = 'Estandarización validada con pasaje de mu_sigma.')


    ###########################################################
    # Predecir calidad
    ###########################################################
    elif (args[0] == "predecir_calidad"):
        if 'funcion' in kwargs:
            predecir_calidad = kwargs['funcion']

        np.random.seed(41)
        N = 8
        d=5; d1=4;d2=3;d3=1
        X = np.random.randn(N,d)
        params = {}
        params['W1'] = np.random.randn(d,d1)
        params['W2'] = np.random.randn(d1,d2)
        params['W3'] = np.random.randn(d2,d3)
        params['b1'] = np.random.randn(d1)
        params['b2'] = np.random.randn(d2)
        params['b3'] = np.random.randn(d3)

        calidad = predecir_calidad(X, params)

        calidad_esperada = np.array([[2.84518388], [2.84518388], [1.97682581], [2.84518388], [2.84518388],
                    [5.59125124], [2.98640787], [3.42680515]])

        fuaa_assert(calidad.shape == calidad_esperada.shape, mensajeFalse="Dimensiones de la salida no validada.", 
                                                              mensajeTrue="Dimensiones de la salida validada.")
        fuaa_assert(error_relativo(calidad, calidad_esperada) < 10e-8, mensajeFalse="Cálculo de la salida no validado.", 
                                                           mensajeTrue="Cálculo de la salida validado.")

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
            
