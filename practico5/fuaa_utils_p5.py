#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 07:52:52 2021

@author: fuaa
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py

_THRESHOLD = 1e-8

############################################################################################
def mostrar_frontera_decision(modelo, X, y):
    plt.figure(figsize=(5,5))
    # Set min and max values and give it some padding
    x_min, x_max = X[:,0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:, 1].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = modelo(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.axis('equal')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    

############################################################################################
def error_relativo(x, y):
    ''' Devuelve el error relativo. '''
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


############################################################################################
def calcular_gradiente_numerico_array(f, x, df, h=1e-5):
    '''
    Evalúa el gradiente numérico para una función, acepta un arreglo numpy
    y devuelve un arreglo numpy.
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


############################################################################################
def calcular_gradiente_numerico(f, x, verbose=True, h=0.00001):
    '''
    Evalúa el gradiente numérico de f en x.

    Entrada:
        f: una función que recibe un solo argumento.
        x: punto (numpy array) en que se evalúa el gradiente.

    Salida:
        grad: gradiente numérico.
    '''

    # se inicializa el gradiente
    grad = np.zeros_like(x)
    # se define un iterador sobre todos los elementos de x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # se evalúa la función en x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h  # se suma h al valor original de x
        fxph = f(x)  # se evalúa f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)  # se evalúa f(x - h)
        x[ix] = oldval  # se restaura el valor original de x

        # se calcula la derivada parcial con la fórmula centrada
        grad[ix] = (fxph - fxmh) / (2 * h)
        if verbose:
            print(ix, grad[ix])
        it.iternext()  # step to next dimension

    return grad

############################################################################################
def generar_flor():
    np.random.seed(1)
    N = 400  # número de muestras
    Nc = int(N/2)  # número de muestras por clase
    D = 2  # dimension de las características
    X = np.zeros((N, D))  # matríz de datos, cada fila es una muestra
    Y = np.zeros((N, 1), dtype='bool')  # vector de etiquetas (0 rojo, 1 azul)
    a = 4  # largo máximo del pétalo

    for j in range(2):
        ix = range(Nc*j, Nc*(j+1))
        t = np.linspace(j*3.12, (j+1)*3.12, Nc) + \
            np.random.randn(Nc)*0.2  # angulo
        r = a*np.sin(4*t) + np.random.randn(Nc)*0.2  # radio
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    return X, Y


############################################################################################
def load_cats_dataset():

    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    # se levantan los datos de entrenamiento
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # las características
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # las etiquetas

    # se levantan los datos de test
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # las características
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # las etiquetas

    classes = np.array(test_dataset["list_classes"][:])  # lista de clases

    train_set_y_orig = train_set_y_orig.reshape(train_set_y_orig.shape[0], 1)
    test_set_y_orig = test_set_y_orig.reshape(test_set_y_orig.shape[0], 1)

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

############################################################################################
def load_spam_dataset():
    X = []
    y = []
    # se leen los datos de la base
    with open('spambase/spambase.data') as f:
        for line in f:
            curr = line.split(',')
            new_curr = []
            for item in curr[:len(curr)-1]:
                new_curr.append(float(item))
            X.append(new_curr)
            y.append([float(curr[-1])])

    X = np.array(X)
    y = np.array(y)
    return X, y


############################################################################################
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


############################################################################################
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


############################################################################################
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



############################################################################################
def validar_parametros( parametros, min_params ):
    """
    Validar número de parámetros.
    """
    condicion = (len( parametros ) >= min_params)
    if not condicion:
        printcolor( "[validar_resultado] Insuficientes parámetros (\"%s\"), se necesitan %d, hay %d." % \
            (parametros[0], min_params, len(parametros)), "r" )
    return condicion
    

############################################################################################
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


############################################################################################
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
    

#########################################################################################
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


#########################################################################################
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
        d0, d1, d2 = 3, 6, 1
        W1_correcto = np.array([[ 0.01624345, -0.00611756, -0.00528172, -0.01072969,  0.00865408, -0.02301539], 
                                [ 0.01744812, -0.00761207,  0.00319039, -0.00249370,  0.01462108, -0.02060141],
                                [-0.00322417, -0.00384054,  0.01133769, -0.01099891, -0.00172428, -0.00877858]])
        b1_correcto = np.array([0., 0., 0., 0., 0., 0.])
        W2_correcto = np.array([[0.00042214], [0.00582815], [-0.01100619],[0.01144724], [0.00901591], [0.00502494]])
        b2_correcto = np.array([0.])

        parametros = test_algoritmo(d0, d1, d2)
        W1 = parametros['W1']
        b1 = parametros['b1']
        W2 = parametros['W2']
        b2 = parametros['b2']

        # Chequear dimensiones.
        fuaa_assert(W1.shape == (d0, d1), mensajeFalse="Dimensiones de W1 no validadas.", mensajeTrue="Dimensiones de W1 validadas.")
        fuaa_assert(b1.shape == (d1, ), mensajeFalse="Dimensiones de b1 no validadas.", mensajeTrue="Dimensiones de b1 validadas.")
        fuaa_assert(W2.shape == (d1, d2), mensajeFalse="Dimensiones de W2 no validadas.", mensajeTrue="Dimensiones de W2 validadas.")
        fuaa_assert(b2.shape == (d2, ), mensajeFalse="Dimensiones de b2 no validadas.", mensajeTrue="Dimensiones de b2 validadas.")

        # Chequear generación de los pesos.
        fuaa_assert(son_iguales(W1, W1_correcto), mensajeFalse="Cálculo de W1 no validado.", mensajeTrue="Cálculo de W1 validado.")
        fuaa_assert(son_iguales(b1, b1_correcto), mensajeFalse="Cálculo de b1 no validado.", mensajeTrue="Cálculo de b1 validado.")
        fuaa_assert(son_iguales(W2, W2_correcto), mensajeFalse="Cálculo de W2 no validado.", mensajeTrue="Cálculo de W2 validado.")
        fuaa_assert(son_iguales(b2, b2_correcto), mensajeFalse="Cálculo de b2 no validado.", mensajeTrue="Cálculo de b2 validado.")

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
        activaciones = ['relu', 'sigmoide','tanh']

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
