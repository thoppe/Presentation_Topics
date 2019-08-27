import tensorflow as tf
import numpy as np
tf.enable_eager_execution()

# 0.6388765649999399 total time for inital condition

def compute_ramp(y):
    x = tf.linspace(0.0,1.0,N)
    g = tf.constant(9.8)

    dx = x[1] - x[0]
    
    v = tf.sqrt(2*g*(y[0]-y))

    dy = y[:-1]-y[1:]
    h = tf.sqrt(dx**2+dy**2)

    cA = 0.5*g*dy/h
    cB = v[:-1]
    cC = -h

    root0 = (-cB - tf.sqrt(cB**2-4*cA*cC))/(2*cA)
    root1 = (-cB + tf.sqrt(cB**2-4*cA*cC))/(2*cA)

    t = tf.where(dy>0, root1, root0)
    #t = tf.where(dy>0, root0, root1)
        
    total_time = tf.reduce_sum(t)

    return total_time

def train_step(Y):
    with tf.GradientTape() as tape:
        time = compute_ramp(Y)

    variables = [Y]
    
    grads = tape.gradient(time, variables)
    optimizer.apply_gradients(zip(grads, variables))

    # Fix the endpoints
    Y[0].assign(1.0)
    Y[-1].assign(0.0)

    return time, y


N = 200

y = np.linspace(1.0, 0.0, N)

y = tf.Variable(y, dtype=tf.float32)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

import pylab as plt

for i in range(1000):
    t, yx = train_step(y)
    print(i, t.numpy())

    if i%100 == 0:
        plt.plot(yx.numpy())
plt.show()

