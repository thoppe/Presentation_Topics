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
    h = tf.sqrt(dx**2+dy**2)

    cA = 0.5*g*dy[0]/h[0]
    t0 = tf.sqrt(4*cA*h[0])/(2*cA)

    # x = x0 + v*t
    # t = -x/v
    t = h/((v[1:]+v[:-1])/2)

    #t2 = tf.sqrt((1+dy**2/dx**2)/(2*g*(y[0]-y)[1:])) * dx
    #print(tf.reduce_sum(t-t2))
    #exit()

    print("H[0]", h[0].numpy(), t0.numpy(), t[0].numpy() - t0.numpy())
    
    total_time = tf.reduce_sum(t)
    

    return total_time

def train_step(Y):
    with tf.GradientTape() as tape:
        time = compute_ramp(Y)

    variables = [Y]   
    grads = tape.gradient(time, variables)

    #print(grads)
    #exit()
    
    optimizer.apply_gradients(zip(grads, variables))

    # Fix the endpoints
    #Y[0].assign(1.0)
    #Y[-1].assign(0.0)

    return time.numpy()

N = 100
y = np.linspace(1.0, 0.0, N+2)[1:-1]
#y = np.random.uniform(-1,0.5,size=(N+2),)[1:-1]

y = tf.Variable(y, dtype=tf.float32)


#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

import pylab as plt

for i in range(5000):
    t = train_step(y)
    print(i, t)
    

    if i%100 == 0 and i>1000:
        yx = np.hstack([1, y.numpy(), 0])
        plt.plot(yx)

        
plt.show()

