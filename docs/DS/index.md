% Title: Stupid Tensorflow tricks II
% Author: Travis Hoppe

!!(figures/oQbLeq4nOek.jpg class="light")

...bg-black
..aligncenter

# .text-landing **Stupid**
# .text-landing **Tensorflow**
# .text-landing **Tricks**
.text-landing **part II**

#### _Brachistochrone problem_
<br><br><br>

by [`@metasemantic`](https://twitter.com/metasemantic)

%% ..text-intro [https://github.com/thoppe/Presentation_Topics/blob/master/docs/DS/Brachistochrone.ipynb](https://github.com/thoppe/Presentation_Topics/blob/master/docs/DS/Brachistochrone.ipynb)

-----

...bg-white
...aligncenter

## wtf? Stupid Tensorflow? _Part II_?
_free automatic differentiation_

#### Part I: Thomson Problem

::medium:: [https://towardsdatascience.com/stupid-tensorflow-tricks-3a837194b7a0](https://towardsdatascience.com/stupid-tensorflow-tricks-3a837194b7a0)
!(figures/1_6wcPkWRp4BYpV_b68sONVw.gif width=50%)

------

...bg-black
...aligncenter

.text-data **Brachistochrone**
</br>
βράχιστος χρόνος: _shortest time_

.align-center !(figures/oldguys.png width=60%)
<br>

Newton, l'Hôpital, at least two Bernoullis . . .

-------

...bg-black
...aligncenter

.text-data **Brachistochrone** </br>
βράχιστος χρόνος: _shortest time_

%% .align-center !(figures/Brachistochrone.gif)
.align-center !(figures/plane.png width=40%)

------
...bg-white
...aligncenter

<style scoped> code { font-size: 150%;  } </style>

# .text-landing **Solve with physics**
_and tensorflow 2.0!_

..wrap
```
import tensorflow as tf
import numpy as np
print(tf.__version__)

# This lets us see computations as soon as we run them!
tf.enable_eager_execution()

> 1.14.0
```
### Conservation of Energy
$$U=K ; mgh = \frac{1}{2}mv^2 $$
$$v = \sqrt{2gh}$$

-----

...wrap
# Time to roll down 
can compute for each little segment...

### $$ z(t) = z_0 + v_0 t + \frac{1}{2}a_z t^2 = z_0 + v_0 t + \frac{g \sin(\theta) t^2}{2} $$
### $$ t = \frac{-v_0 \pm \sqrt{v_0^2 - 2 g \sin(\theta)z_0}} {g \sin(\theta)} $$


-----
...aligncenter
..wrap
# _Ugh._ Can we be lazy?
Over a short enough time span we can ignore gravity, as long as we fix the velocity each segment. $ z(t) = z_0 + v_0 t ;  t = z_0 / v_0  $
#### $$ t = \frac{\sqrt{\Delta x^2 + \Delta y^2}}{v_0}  $$
!(figures/plane_crop.png width=30%)

-----

...bg-white
...aligncenter

## Setup the "ramp"

..aligncenter
``` 
N = 20
width = 3
y = tf.Variable(np.linspace(0.99, 0.01, N,), dtype=tf.float32)
x = tf.linspace(0.0, 1.0, N+2) * width
print(y)

<tf.Variable 'Variable:0' shape=(20,) dtype=float32, numpy=
array([0.99      , 0.9384211 , 0.88684213, 0.83526313, 0.7836842 ,
       0.73210526, 0.6805263 , 0.6289474 , 0.57736844, 0.5257895 ,
       0.47421053, 0.4226316 , 0.37105262, 0.31947368, 0.26789474,
       0.21631579, 0.16473684, 0.1131579 , 0.06157895, 0.01      ],
      dtype=float32)>
```


------

...bg-white
...aligncenter

...wrap

# Objective function
..aligncenter
```
def compute_ramp(y):
    g = tf.constant(9.8)
    
    # Fix the endpoints
    y = tf.concat([[1.0], y, [0.0]], axis=0)
    
    # All the widths are the same
    dx = x[1] - x[0]
    dy = y[:-1] - y[1:]   
    dz = tf.sqrt(dx ** 2 + dy ** 2)

    # Conservation of energy
    v = tf.sqrt(2 * g * (y[0] - y))
    
    # The average of the starting and ending velocity at each segment
    avg_v = (v[1:] + v[:-1]) / 2

    total_time = tf.reduce_sum(dz / avg_v)
    return total_time

```

------

...bg-white
...aligncenter

...wrap

# Training step, different in tf 2.0!

..aligncenter
```
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

def train_step(y):
    with tf.GradientTape() as tape:
        time = compute_ramp(y)

    variables = [y]
    grads = tape.gradient(time, variables)
    optimizer.apply_gradients(zip(grads, variables))

    return time.numpy()
```

------
...aligncenter
..wrap

# Plot convergence
full code in notebook

!(figures/final_solution.png width=60%)

------
...aligncenter
..wrap
# Did it work? Yes!
Wikipedia's solution below

!(figures/Brachistochrone.gif width=60%)

------

...bg-black
...aligncenter

# Yeah, so?

+ Auto-diff programs like Tensorflow (eg. torch) can do more than deep learning.

# Ok, so all the things?

+ Works if your problem can be made smooth.

# I don't want to think too hard.
+ On short enough scales, everything is flat.

-----

..wrap

## **Thanks, you!**
#### Comment at
## [@metasemantic](https://twitter.com/metasemantic?lang=en)
#### Repo at
..aligncenter
!(figures/qrcode.png width=20%)





