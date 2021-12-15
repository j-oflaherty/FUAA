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


#########################################################################
def split_data(all_data):
    '''
    Separa los datos en 10 Folds predefinidos
    '''
    folders = [[], [], [], [], [], [], [], [], [], []]
    for ln in range(len(all_data)):
        folders[int(all_data[ln][5]) - 1].append(all_data[ln])
    for ln in range(len(folders)):
        folders[ln] = np.array(folders[ln])
    return np.array(folders, dtype=object)


#########################################################################
def mostrar_senhales(path_audios, n_fold, n_clase):
    '''
    Muestra las señales de la clase n_clase almacenadas en el fold n_fold
    Entrada:
      path_audios: directorio donde se almacena la base
      n_fold: fold a utilizar
      n_clase: número de clase a mostrar
    '''
    cont = 0
    folder = os.path.join(path_audios, 'fold' + str(n_fold), str(n_clase))
    tam = len(glob.glob(os.path.join(folder, '*.npy')))
    fig, axes = plt.subplots(tam, 1, figsize=_FIGSIZE)
    for path_au in glob.glob(os.path.join(folder, '*.npy')):
        y = np.load(path_au)
        axes[cont].plot(np.arange(len(y)), y)
        axes[cont].grid()
        axes[cont].axis([0, len(y), -1.2, 1.2])
        #axes[cont].set_title(path_au)
        cont += 1
    axes[0].set_title('fold %d, clase %d' % (n_fold, n_clase))
    fig.tight_layout()


#########################################################################
def comparar_dos_clases(path_audios, fold1, clase1, fold2, clase2):
    '''
    Entrada:
      path_audios: directorio donde se almacena la base
      fold1: fold donde se encuentran las primeras señales a comparar
      clase1: clase de las primeras señales a comparar
      fold2: fold donde se encuentran las segundas señales a comparar
      clase2: clase de las segundas señales a comparar
    '''
    path1 = os.path.join(path_audios, 'fold' + str(fold1), str(clase1),
                         '*.npy')
    path2 = os.path.join(path_audios, 'fold' + str(fold2), str(clase2),
                         '*.npy')
    tam_clas_mayor = np.max([len(glob.glob(path1)), len(glob.glob(path2))])
    fig, axes = plt.subplots(nrows=tam_clas_mayor, ncols=2, figsize=(15, 10))
    for fold, clase in zip([fold1, fold2], [clase1, clase2]):
        cont = 0
        path = os.path.join(path_audios, 'fold' + str(fold), str(clase),
                            '*.npy')
        for path_au in glob.glob(path):
            y = np.load(path_au)
            if clase == clase1:
                axes[cont, 0].plot(np.arange(len(y)), y)
                axes[cont, 0].grid()
                axes[cont, 0].axis([0, len(y), -1.2, 1.2])
            else:
                axes[cont, 1].plot(np.arange(len(y)), y)
                axes[cont, 1].grid()
                axes[cont, 1].axis([0, len(y), -1.2, 1.2])
            cont += 1
        if clase == clase1:
            axes[0, 0].set_title('fold %d, clase %d' % (fold, clase))
        else:
            axes[0, 1].set_title('fold %d, clase %d' % (fold, clase))
    fig.tight_layout()


#########################################################################
def features_from_arrays(features_from_fold, tipo_c1='', tipo_c2=''):
    '''
    Entrada:
        features_from_fold: características calculadas en el fold
        tipo_c1: string que puede ser mean, min, max, std o vacio para calcular el ZCR
        tipo_c2: string que puede ser mean, min, max, std o vacio para calcular el RMS

    Salida:
        numpy array con dos características:
            ZCR: ratio de cruces por cero, en el caso de tener entrada vacia se devuelve el array entero
            RMS: raíz del error cuadrático medio, en el caso de tener entrada vacia se devuelve el array entero
    '''
    for i in range(len(features_from_fold)):
        if tipo_c1 == 'mean':
            features_from_fold[i][0] = np.mean(features_from_fold[i][0])
        elif tipo_c1 == 'min':
            features_from_fold[i][0] = np.min(features_from_fold[i][0])
        elif tipo_c1 == 'max':
            features_from_fold[i][0] = np.max(features_from_fold[i][0])
        elif tipo_c1 == 'std':
            features_from_fold[i][0] = np.std(features_from_fold[i][0])

        if tipo_c2 == 'mean':
            features_from_fold[i][1] = np.mean(features_from_fold[i][1])
        elif tipo_c2 == 'min':
            features_from_fold[i][1] = np.min(features_from_fold[i][1])
        elif tipo_c2 == 'max':
            features_from_fold[i][1] = np.max(features_from_fold[i][1])
        elif tipo_c2 == 'std':
            features_from_fold[i][1] = np.std(features_from_fold[i][1])

    return features_from_fold


#########################################################################
def plot_features(features, labels, op_type1_1, op_type2_1, objects):
    
    plt.figure(figsize=_FIGSIZE)
    clases = np.unique(labels).astype(int)
    for c in clases:
        indices = labels==c
        plt.scatter(features[indices,0], features[indices,1], label=objects[c])
    plt.grid()
    plt.legend()
    plt.xlabel('ZCR ' + op_type1_1)
    plt.ylabel('RMS ' + op_type2_1)
    plt.title('Características calculadas')


#########################################################################
def generar_conjunto_entrenamiento(folds, clases, op_feat1, op_feat2, folders_features, folders, objects):
    '''
    Entrada:
        folds: folds a utilizar
        clases: clases a utilizar
        op_feat1: operación a realizar con los valores ZCR del segmento
        op_feat1: operación a realizar con los valores RMS del segmento
    Salida:
        features: características calculadas
        labels: etiquetas asociadas a las caracteríticas
    '''
    features = []
    labels = []
    for c in range(len(clases)):
        for fold in folds:
            f_c = folders_features[fold-1][folders[fold-1][:, -2] == np.str(clases[c])]
            f_c = features_from_arrays(f_c, tipo_c1=op_feat1, tipo_c2=op_feat2)
            features.append(f_c)
            labels.append(clases[c] * np.ones(len(f_c)))
    features = np.vstack(np.asarray(features))
    labels = np.concatenate(np.asarray(labels))
    
    plot_features(features, labels, op_feat1, op_feat2, objects)
    return features, labels


#########################################################################
def mostrar_superficie_decision(features, labels, clf_svm, clf_logreg, objects):
    ''' 
    función auxiliar para mostrar el resultado de la clasificación 
    '''
    
    plt.figure(figsize=_FIGSIZE)
    clases = np.unique(labels).astype(int)
    for c in clases:
        indices = labels==c
        plt.scatter(features[indices,0], features[indices,1], label=objects[c])

    # plot the decision svm
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf_svm.decision_function(xy).reshape(XX.shape)
    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(clf_svm.support_vectors_[:, 0], clf_svm.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k', label='vectores de soporte')

    # plot regresión logística
    w_log = clf_logreg.coef_[0]
    b_log = clf_logreg.intercept_[0]
    #ap = -w_log[0] / w_log[1]
    #ymin = features[:, 1].min()
    #ymax = features[:, 1].max()
    #if ap >= 0:
    #    xmin = np.max([(ymin + b_log / w_log[1]) / ap, features[:, 0].min()])
    #    xmax = np.min([(ymax + b_log / w_log[1]) / ap, features[:, 0].max()])
    #else:
    #    xmin = np.max([(ymax + b_log / w_log[1]) / ap, features[:, 0].min()])
    #    xmax = np.min([(ymin + b_log / w_log[1]) / ap, features[:, 0].max()])
    xmin = features[:, 0].min()
    xmax = features[:, 0].max()
    xx = np.linspace(xmin, xmax)
    yy = -( w_log[0] * xx + b_log) / w_log[1]
    plt.plot(xx, yy, 'r-', label='límite de decisión de regresión logística')
    
    plt.legend()
    plt.xlabel('ZCR ')
    plt.ylabel('RMS ')
    plt.grid()
    plt.show()


#########################################################################
def plot_svm_multiclase(X, Y, clf, op_type1_1, op_type2_1):

    plt.figure(figsize=_FIGSIZE)

    # plot the decision
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired, alpha=.1, shading='auto')
    # plot support vectors
    plt.scatter(clf.support_vectors_[:, 0],
                clf.support_vectors_[:, 1],
                s=100,
                linewidth=1,
                facecolors='none',
                edgecolors='k',
                label='support vectors')

    # plt.legend()
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors='k')
    plt.xlabel('ZCR ' + op_type1_1)
    plt.ylabel('RMS ' + op_type2_1)
    plt.grid()
    plt.show()


#########################################################################
def extraer_caracteristicas(folds_list):
    '''
    Función que parsea los folds para generar conjuntos de entrenamiento/validación/test
    Entrada:
        folds_list: lista que contiene los folds que se quieren utilizar 
                    para construir el conjunto
    Salida:
        features: arreglo de tamaño (N, 275) que contiene las características extraídas 
                  de los folds
        labels: vector de tamaño (N,) que contiene las clases a la que pertenecen
                las muestras
        fold_indices: vector de tamaño (N,) que contiene a que fold pertenece la muestra
    '''
    data_list = []  # lista auxiliar que contiene los folds
    indices_list = []  # lista auxiliar que contiene los indices de los folds
    for foldNumber in folds_list:
        filename = 'dataset/' + "features_fold_" + str(foldNumber) + ".csv"
        data = np.loadtxt(filename, delimiter=',',
                          skiprows=1)  # se lee un fold
        indices = foldNumber * np.ones(
            data.shape[0])  # se asocia un número de fold a cada dato
        data_list.append(data)  # se agrega la data del fold a la lista
        indices_list.append(
            indices)  # se agregan los indices a la lista de indices
    data = np.concatenate(data_list)
    fold_indices = np.concatenate(indices_list)
    features = data[:, :275]
    labels = data[:, 276]
    return features, labels, fold_indices


#########################################################################
def mostrar_grid_search_results(rangoC, rangoGamma, resultados):
    '''
    Función auxiliar que muestra los parámetros del gridsearch.
    Entrada:
        rangoC: lista con los valores de C evaluados
        rangoGamma: lista con los valores de gamma evaluados
        resultados: arrglo de tamaño (len(rangoC),len(rangoGamma)) con los resultados
                    del grid-search
    '''
    fig, ax = plt.subplots()
    im = ax.imshow(resultados, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_ylabel('C')
    ax.set_ylim(-0.5,len(rangoC)-0.5)
    ax.set_xlim(-0.5,len(rangoGamma)-0.5)
    ax.set_yticks(range(len(rangoC)))
    ax.set_xticks(range(len(rangoGamma)))
    rangoC_labels = [str(c) for c in rangoC]
    ax.set_yticklabels(rangoC_labels)
    ax.set_xlabel('Gamma')
    rangoGamma_labels = [str(gamma) for gamma in rangoGamma]
    ax.set_xticklabels(rangoGamma_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                    rotation_mode="anchor")
    plt.title('Resultados del Grid Search')


#########################################################################
def mostrar_matriz_confusion(cm, target_names,
                            normalize=False,
                            cmap=plt.cm.Blues):
    '''
    Muestra la matriz de confusión
    '''

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=target_names, yticklabels=target_names,
            title= 'Matriz de confusion normalizada' if normalize else 'Matriz de confusion sin normalizar',
            ylabel='Etiqueta',
            xlabel='Predicción',
            xlim=(-0.5,9.5),
            ylim=(9.5,-0.5))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
###########################################################################

def plotLine(ax, xRange, w, x0, label, color='grey', linestyle='-', alpha=1.):
    """ Plot a (separating) line given the normal vector (weights) and point of intercept """
    if type(x0) == int or type(x0) == float or type(x0) == np.float64:
        x0 = [0, -x0 / w[1]]
    yy = -(w[0] / w[1]) * (xRange - x0[0]) + x0[1]
    ax.plot(xRange, yy, color=color, label=label, linestyle=linestyle)
    
def plotSvm(X, y, support=None, support_labels=None, w=None, b=0., label='Datos', separatorLabel='Limite de decisión', 
            ax=None, bound=[[-1., 1.], [-1., 1.]], title='SVM'):
    """ Muestra el límite de decisión SVM y el margen """
    if ax is None:
        fig, ax = plt.subplots(1)
    

    colors = ['blue','red']
    cmap = pltcolors.ListedColormap(colors)
    im = ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap, alpha=0.5, label=label)
    if support is not None:
        ax.scatter(support[:,0], support[:,1], label='Support', s=80, facecolors='none', 
                   edgecolors='y', color='y')
        #print("Número de vectores de soporte = %d" % (len(support)))
    if w is not None:
        xx = np.linspace(bound[0][0],bound[0][1], 100) #np.array(bound[0])
        plotLine(ax, xx, w, b, separatorLabel)
        # Plot margin
        if support is not None:
            # margin = 2 / np.sqrt(np.dot(w, w))
            # signed_dist = support @ w  + intercept
            # min_support = np.min(signed_dist)
            # max_support = np.max(signed_dist)
            #print('min, max :',  min_support, max_support)
            #print('margin :',  margin)
            #print('w', w)
            plotLine(ax, xx, w, (b + 1), 'Margen -', linestyle='-.', alpha=0.8)
            plotLine(ax, xx, w, (b - 1), 'Margen +', linestyle='--', alpha=0.8)
            ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid()
    ax.set_xlim(bound[0])
    ax.set_ylim(bound[1])
    cb = plt.colorbar(im, ax=ax)
    loc = np.arange(-1,1,1)
    cb.set_ticks(loc)
    cb.set_ticklabels(['-1','1'])
