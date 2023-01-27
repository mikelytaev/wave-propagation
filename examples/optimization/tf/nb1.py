import numpy as np
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
import tensorflow_probability as tfp
import math as fm


x = tf.Variable([0.1j, 0.1j], tf.complex128)
y = tf.Variable([0.1j, 0.1j], tf.complex128)

rational_order = (7, 7)
num_coef = tf.reshape(tf.Variable([0.1j] * rational_order[0], tf.complex128), (rational_order[0], 1))
den_coef = tf.reshape(tf.Variable([0.1j] * rational_order[1], tf.complex128), (rational_order[1], 1))
k0 = 2 * fm.pi
dx = 10
dz = 0.5
theta_max_degrees = 10
k_z_max = k0 * fm.sin(theta_max_degrees)
k_z_grid = tf.reshape(tf.constant(np.linspace(0, k_z_max, 100)), (1, 100))
loss_fn = lambda: (tf.math.log(tf.norm(x)) + 2*tf.norm(y) - 5.)**2

loss_fn2 = lambda: loss_fn() + 2*loss_fn()

t_i = lambda k_z, a_i, b_i: (1 - 4*a_i/(k0*dz)**2*tf.math.sin(k_z*dz/2)**2) / (1 - 4*b_i/(k0*dz)**2*tf.math.sin(k_z*dz/2)**2)
de_numerator = lambda a: tf.reduce_sum(tf.math.log((1 - 4*a/(k0*dz)**2*tf.math.sin(k_z_grid*dz/2)**2)), axis=0)

discrete_k_x = lambda k_z: k0 + (de_numerator(k_z, num_coef) - de_numerator(k_z, den_coef)) / (1j * dx)
#loss_fn3 = lambda k_z:

losses = tfp.math.minimize(loss_fn2,
                           num_steps=200,
                           optimizer=tf.optimizers.Adam(learning_rate=0.1))

print("optimized value is {} with loss {}".format(x, losses[-1]))