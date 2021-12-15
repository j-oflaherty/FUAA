import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy.io

def mostrar_frontera_decision(modelo, X, y):
    plt.figure(figsize=(10,10))
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
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.5)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    


def generar_flor():
    np.random.seed(1)
    N = 400 # número de muestras
    Nc = int(N/2) # número de muestras por clase
    D = 2 # dimension de las características
    X = np.zeros((N,D)) # matríz de datos, cada fila es una muestra
    Y = np.zeros((N,1), dtype='bool') # vector de etiquetas (0 rojo, 1 azul)
    a = 4 # largo máximo del pétalo 

    for j in range(2):
        ix = range(Nc*j,Nc*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,Nc) + np.random.randn(Nc)*0.2 # angulo
        r = a*np.sin(4*t) + np.random.randn(Nc)*0.2 # radio
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    return X, Y


def load_cats_dataset():

    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    # se levantan los datos de entrenamiento
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # las características 
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # las etiquetas

    # se levantan los datos de test
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # las características
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # las etiquetas

    classes = np.array(test_dataset["list_classes"][:]) # lista de clases
    
    train_set_y_orig = train_set_y_orig.reshape(train_set_y_orig.shape[0],1)
    test_set_y_orig = test_set_y_orig.reshape(test_set_y_orig.shape[0], 1)

    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def load_spam_dataset():
    X=[]; y=[]
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

def load_2D_dataset():
    data = scipy.io.loadmat('data.mat')
    train_X = data['X']
    train_Y = data['y']
    test_X = data['Xval']
    test_Y = data['yval']
    
    return train_X, train_Y, test_X, test_Y


