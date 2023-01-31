import numpy as np
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
import tensorflow_probability as tfp
import math as fm




rational_order = (7, 7)
num_coef = tf.Variable(tf.reshape(tf.Variable((np.random.rand(rational_order[0])+np.random.rand(rational_order[0])*1j - 0.5 - 0.5j)*1, tf.complex128), (rational_order[0], 1)))
den_coef = tf.Variable(tf.reshape(tf.Variable((np.random.rand(rational_order[1])+np.random.rand(rational_order[1])*1j - 0.5 - 0.5j)*1, tf.complex128), (rational_order[1], 1)))
k0 = 2 * fm.pi
dx = 50
dz = 0.25
theta_max_degrees = 22
k_z_max = k0 * fm.sin(theta_max_degrees*fm.pi/180)
k_z_grid = np.linspace(0, k_z_max, 100)+0j
k_z_grid_tf = tf.reshape(tf.constant(k_z_grid, tf.complex128), (1, len(k_z_grid)))
d_k_z = k_z_grid[1] - k_z_grid[0]


de_numerator = lambda a: tf.reduce_sum(tf.math.log((1 - 4 * a / (k0*dz) ** 2 * tf.math.sin(k_z_grid_tf * dz / 2) ** 2)), axis=0)

discrete_k_x = lambda: k0 + (de_numerator(num_coef) - de_numerator(den_coef)) / (1j * dx)
loss_fn3 = lambda: tf.abs(tf.sqrt(tf.reduce_sum((discrete_k_x() - tf.math.sqrt(k0 ** 2 - k_z_grid_tf ** 2)) ** 2, axis=1)) * d_k_z)
loss_fn4 = lambda: tf.reduce_max(tf.abs(discrete_k_x() - tf.math.sqrt(k0 ** 2 - k_z_grid_tf ** 2)))

trace_fn = lambda traceable_quantities: {
  'loss': traceable_quantities.loss, 'num_coef': num_coef}
losses = tfp.math.minimize(loss_fn4,
                           num_steps=100000,
                           optimizer=tf.optimizers.Adam(learning_rate=10, epsilon=0.1),
                           trainable_variables=[num_coef, den_coef],
                           #trace_fn=trace_fn
                           )

print("optimized value is {} with loss {}".format(num_coef, losses[-1]))

# optimizer = tf.optimizers.SGD(learning_rate=10, nesterov=True)
#
# for i in range(20000):
#   with tf.GradientTape() as tape:
#     loss = loss_fn4()
#   print(loss)
#   grads = tape.gradient(loss, [num_coef, den_coef])
#   print(num_coef)
#   optimizer.apply_gradients(zip(grads, [num_coef, den_coef]))
#
# print("Minimum value: ", loss_fn4())