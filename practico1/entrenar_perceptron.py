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
    
    cantidad_de_muestras = X.shape[0]

    iteraciones = 0
    hay_mal_clasificadas = True
    error = list()
    while (iteraciones < max_iter) and (hay_mal_clasificadas):
        res_y = np.sign(X @ w)
        err = y != res_y
        error.append((1/cantidad_de_muestras)*np.count_nonzero(err))
        hay_mal_clasificadas = np.count_nonzero(err) > 0
        if hay_mal_clasificadas:
            w = w + y[err][0] * X[err][0]
        iteraciones += 1

    #######################################################
    ######## TERMINA ESPACIO PARA COMPLETAR CODIGO ########
    #######################################################
    
    return w, error
