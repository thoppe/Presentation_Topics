import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

width = 3
# 1.4031373262405396 total time for inital condition
# 1.0077402591705322 best time


def compute_ramp(y):
    g = tf.constant(9.8)

    dx = x[1] - x[0]

    # Fix the endpoints
    y = tf.concat([[1.0], y, [0.0]], axis=0)

    # Conservation of energy U=K, mgh=(1/2)mv**2
    v = tf.sqrt(2 * g * (y[0] - y))

    dy = y[:-1] - y[1:]
    avg_v = (v[1:] + v[:-1]) / 2

    h = tf.sqrt(dx ** 2 + dy ** 2)

    total_time = tf.reduce_sum(h / avg_v)
    return total_time


def train_step(y):
    with tf.GradientTape() as tape:
        time = compute_ramp(y)

    variables = [y]
    grads = tape.gradient(time, variables)
    optimizer.apply_gradients(zip(grads, variables))

    return time.numpy()


N = 500
n_steps = 5000

#y = np.linspace(0.5, 0.01, N)
y = np.random.uniform(-1,0.5,size=(N,),)

y = tf.Variable(y, dtype=tf.float32)
x = tf.linspace(0.0, 1.0, N + 2) * width

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
print(f"Starting time {compute_ramp(y)}")

import pylab as plt
import seaborn as sns


plt.figure(figsize=(7,4))
for i in range(n_steps):
    t = train_step(y)

    if i % 100 == 0 and i >= 200:
        print(f"{i} {t:0.16f}")
        yx = np.hstack([[1], y.numpy(), [0]])
        plt.plot(x.numpy(), yx, color='b',alpha=i/n_steps)

plt.plot(x.numpy(), yx,'r',lw=1)
plt.plot(x.numpy(), 0*x.numpy(),'k',lw=2)
sns.despine()
plt.tight_layout()
    
plt.show()
