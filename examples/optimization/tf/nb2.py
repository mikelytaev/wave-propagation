import tensorflow as tf

def f(x):
  return tf.sin(tf.math.log(x**2) - x) / x

x = tf.Variable(1.0)
optimizer = tf.optimizers.SGD(learning_rate=0.1)

for i in range(100):
  with tf.GradientTape() as tape:
    loss = f(x)
  grads = tape.gradient(loss, [x])
  optimizer.apply_gradients(zip(grads, [x]))

print("Minimum value: ", x.numpy())