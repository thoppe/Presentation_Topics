import tensorflow as tf
import numpy as np
tf.enable_eager_execution()

# 0.6388765 total time for inital condition
# 0.5835300 current best time

def compute_ramp(y, width=3):
    
    x = tf.linspace(0.0,1.0,N+2)*width
    g = tf.constant(9.8)

    dx = x[1] - x[0]

    # Fix the endpoints
    y = tf.concat([[1.], y, [0.]], axis=0)   
    v = tf.sqrt(2*g*(y[0]-y))
    
    dy = y[:-1]-y[1:]
    avg_v = (v[1:]+v[:-1])/2
    
    h = tf.sqrt(dx**2+dy**2)
    
    total_time = tf.reduce_sum(h/avg_v)
    return total_time

def train_step(Y):
    with tf.GradientTape() as tape:
        time = compute_ramp(Y)

    variables = [Y]   
    grads = tape.gradient(time, variables)
    optimizer.apply_gradients(zip(grads, variables))

    return time.numpy()

N = 100
y = np.linspace(1.0, 0.0, N)[1:-1]
#y = np.random.uniform(-1,0.5,size=(N+2),)[1:-1]

y = tf.Variable(y, dtype=tf.float32)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

import pylab as plt

for i in range(5000):
    t = train_step(y)
    print(i, t)
    

    if i%100 == 0 and i>1000:
        yx = np.hstack([1, y.numpy(), 0])
        plt.plot(yx)

        
plt.show()

