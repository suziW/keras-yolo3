import tensorflow as tf
from keras import backend as K

grid_y = K.tile(K.reshape(K.arange(0, stop=13), [-1, 1, 1, 1]),  #   [?, ?, 1, 1]
    [1, 13, 1, 1])
grid_x = K.tile(K.reshape(K.arange(0, stop=13), [1, -1, 1, 1]),  # [?, ?, 1, 1]
    [13, 1, 1, 1])
grid = K.concatenate([grid_x, grid_y])  # [?, ?, 1, 2]

with tf.Session() as sess:
    g, gx, gy = sess.run([grid, grid_x, grid_y])
    print('==================', g.shape)
    print(g)
    print('=================', gx.shape)
    print(gx)
    print('======================', gy.shape)
    print(gy)