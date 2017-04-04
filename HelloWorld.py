import tensorflow as tf

a = tf.placeholder(tf.float32)
b = a*2

with tf.Session() as session:
    result = session.run(b, feed_dict={a: 3.5})
    print(result)