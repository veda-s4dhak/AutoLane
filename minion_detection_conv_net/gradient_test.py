import tensorflow as tf

if __name__ == '__main__':
    x = tf.placeholder(tf.float32)
    z = 2 * x * x
    y = 6*z
    var_grad = tf.gradients(y,x)
    sess = tf.Session()
    y_p, vg_p = sess.run([y,var_grad],feed_dict={x:1})
    print(y_p)
    print(vg_p)