import numpy as np
from numba import njit
import tensorflow as tf
from tensorflow import keras

extent = 2
resolution = 400
line = np.linspace(-extent, extent, resolution)
grid = np.meshgrid(line, line)
xpts = np.vstack([grid[0].ravel(), grid[1].ravel()]).T

np.warnings.filterwarnings("ignore")

@njit(parallel=True)
def complex_equation(Z, C):
    return Z**2 + C


def generate_set(N=1000,k=100):
    X = np.random.uniform(-2,2,size=(N,2))
    C = X[:,0] + X[:,1]*1J

    Z = np.zeros_like(C)
    for _ in range(k):
        Z = complex_equation(Z, C)
    
    y = ~(np.isnan(Z) | np.isinf(Z))
    y = y.astype(float)*2 - 1

    return X, y


width = 32*2

layers = [
    keras.layers.Dense(width, activation=tf.nn.tanh, input_shape=(2,)),
]

for n in range(10):
    layers.append(keras.layers.Dense(width, activation=tf.nn.tanh))
    
layers.append(keras.layers.Dense(1, activation=tf.nn.tanh))

clf = keras.Sequential(layers)
clf.compile(optimizer='adam', loss='mse')


for n in range(10):
    X, y = generate_set(N=10**5)
    clf.fit(X, y, epochs=1, batch_size=64)



#X, y = generate_set(N=10**4)
yp = clf.predict(xpts,batch_size=2*512).ravel()
yp = (1 + np.clip(yp, -1, 1))/2
yp = yp.reshape((resolution, resolution, 1))


import pixelhouse as ph
canvas = ph.Canvas(resolution, resolution)
canvas.img[:, :, :3] = (yp*255).astype(np.uint8)
canvas.show()
