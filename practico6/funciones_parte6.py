def gram_Q(X, y):
    '''
    Calcula la matriz de Gram Q.
    Entrada: 
        X: matriz de tamaño (N.d)
        y: arreglo unidimensional de tamaño (N)
    Salida:
        Q: arreglo de tamaño (N,N)
    '''
    
    #### EMPIEZA ESPACIO PARA COMPLETAR CODIGO ############
    
    N = X.shape[0]
    
    # Qij = yi * yj * xi * xj
    gram_Q = np.empty((N, N))
    for i in range(0, N):
        for j in range(0, N):
            gram_Q[i][j] = y[i] * y[j] * (X[i] @ X[j])
    
    #### TERMINA ESPACIO PARA COMPLETAR CODIGO ############
    
    return gram_Q

def funcion_dual(alpha, gram_Q):
    '''
    Función a optimizar al resolver el problema dual
    Entrada:
        alpha: arreglo unidimensional de tamaño (N,) 
        gram_Q: matriz de Gram de tamaño (N,N)
    Salida:
        d: valor de la función de costo del problema dual
    '''
    #### EMPIEZA ESPACIO PARA COMPLETAR CODIGO ############
    
    d = 0.5* alpha @ gram_Q @ alpha - np.sum(alpha)
    
    #### TERMINA ESPACIO PARA COMPLETAR CODIGO ############
    
    return d


def restriccion_igualdad(alpha, y):
    '''
    Entrada:
        alpha: arreglo unidimensional de tamaño (N,) 
        y: arreglo unidimensional de tamaño (N,) con las etiquetas
    Salida:
        cero: escalar que tiene que ser igual a cero
    '''
    
    #### EMPIEZA ESPACIO PARA COMPLETAR CODIGO ############
 
    cero = y @ alpha

    #### TERMINA ESPACIO PARA COMPLETAR CODIGO ############
    
    return cero

def restriccion_desigualdad(alpha):
    '''
    Entrada:
        alpha: arreglo unidimensional de tamaño (N,) 
    Salida:
        mayor_que_cero: cantidad que tiene que ser mayor_que_cero
    '''
    
    
    #### EMPIEZA ESPACIO PARA COMPLETAR CODIGO ############
    
    mayor_que_cero = alpha > 0
    
    #### TERMINA ESPACIO PARA COMPLETAR CODIGO ############

    return mayor_que_cero 

def jacobiano_funcion_dual(alpha, gram_Q):
    '''
    Jacobiano de la función a optimizar al resolver el problema dual
    Entrada:
        alpha: arreglo unidimensional de tamaño (N,) 
        gram_Q: matriz de Gram de tamaño (N,N)
    Salida:
        J: arreglo de (N,) con el gradiente de la función dual respecto a alpha
    '''
    #### EMPIEZA ESPACIO PARA COMPLETAR CODIGO ############
    
    N = alpha.shape[0]
    
    #d(Ein)/d(ai) = sum(ajQij) - 1
    J = gram_Q @ alpha - np.ones(N)
    #### TERMIN ESPACIO PARA COMPLETAR CODIGO ############
    
    return J

def jacobiano_restriccion_igualdad(alpha, y):
    '''
    Entrada:
        alpha: arreglo unidimensional de tamaño (N,) 
        y: arreglo unidimensional de tamaño (N,) con las etiquetas
    Salida:
        jacobiano_cero: arreglo unidimensional de tamaño (N,) con el gradiente
                        de la restricción respecto a alpha
    '''
    #### EMPIEZA ESPACIO PARA COMPLETAR CODIGO ############

    N = alpha.shape[0]
    # grad(sum(yi*a))
    
    jacobiano_cero = y 
            
    #### TERMINA ESPACIO PARA COMPLETAR CODIGO ############
    
    return jacobiano_cero

def jacobiano_restriccion_desigualdad(alpha):
    '''
    Entrada:
        alpha: arreglo unidimensional de tamaño (N,) 
    Salida:
        jacobiano_desigualdad: arreglo de tamaño (N, N) con el jacobiano
                               de las restricciones de desigualdad respecto a alpha
    '''
    
    #### EMPIEZA ESPACIO PARA COMPLETAR CODIGO ############
    
    N = alpha.shape[0]
    jacobiano_desigualdad = np.identity(N)
    
    #### TERMINA ESPACIO PARA COMPLETAR CODIGO ############
    
    return jacobiano_desigualdad  


def fit_svm_linealmente_separable(X, y, max_iter=10000, disp=True):
    '''
    Resuelve el problema de optimización dual utilizando el método SLSQP de optimize.minimize() de scipy
    Entrada:
        X: arreglo de tamaño (N,d) con los features
        y: arreglo de tamaño (N) con las etiquetas
        max_iter: número máximo de iteraciones a utilizar en la optimización de la función dual
        disp: si vale True muestra información relativa a la optimización que se está realizando
    Salida:
        alpha: valores de alpha encontrados al optimizar 
    ''' 

    N = len(y)
    Q = gram_Q(X, y)
    alpha_0 = np.ones(N)
    constraints = ({'type': 'eq',   'fun': lambda a: restriccion_igualdad(a, y), 'jac': lambda a: jacobiano_restriccion_igualdad(a,y)},
                   {'type': 'ineq', 'fun': lambda a: restriccion_desigualdad(a), 'jac': lambda a: jacobiano_restriccion_desigualdad(a)})
    opt_result = optimize.minimize(fun=lambda a: funcion_dual(a, Q),
                               x0=alpha_0, 
                               method='SLSQP', 
                               jac=lambda a: jacobiano_funcion_dual(a, Q), 
                               constraints=constraints,
                               options={'maxiter': 10000, 'disp': disp})
    
    alpha = opt_result.x
    
    return alpha


def info_svm_linealmente_separable(X, y, alpha):
    '''
    Entrada:
        X: arreglo de tamaño (N,d) con los features
        y: arreglo de tamaño (N) con las etiquetas
        alpha: arreglo de tamaño (N) con los valores de alpha encontrados al optimizar
    Salida:
        w: coeficientes del modelo svm
        b: término de bias del modelo svm
        vectores_soporte:  muestras de X con valores de alpha_i mayores que epsilon
        etiquetas_soporte: etiquetas asociadas a los vectores de soporte
    '''
    # Valores de alpha_i mayores que epsilon se considerarán distintos de cero
    epsilon = 1e-6

    #### EMPIEZA ESPACIO PARA COMPLETAR CODIGO ############

    w = (alpha * y) @ X

    indices_soporte = alpha > 0

    b = y[indices_soporte][0] - X[indices_soporte][0] @ w

    # Se guardan los vectores de soporte
    vectores_soporte = X[indices_soporte]

    # Se guardan las etiquetas de los vectores de soporte
    etiquetas_soporte = y[indices_soporte]

    #### TERMINA ESPACIO PARA COMPLETAR CODIGO ############

    return w, b, vectores_soporte, etiquetas_soporte


