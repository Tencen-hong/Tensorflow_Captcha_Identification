# coding=utf-8
import tensorflow as tf


class Model(object):
    def __init__(self, dropout, learning_rate, MAX_CAPTCHA_LEN, CHAR_SET_LEN):
       # with tf.name_scope('inputs'):
        self.x = tf.placeholder(tf.float32, [None, 60, 160, 1])
        self.y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA_LEN * CHAR_SET_LEN])
       # with tf.name_scope("LAYER"):
           # with tf.name_scope("layer1"):
                #with tf.name_scope("conv1"):
        # 第一层卷积
        conv1 = tf.layers.conv2d(
            inputs=self.x,
            filters=32,  # 输出空间的维数（卷积过滤器的数量）
            kernel_size=[3, 3],
            strides=1,
            padding='same',
            activation=tf.nn.relu,
        )
                #with tf.name_scope("pool1"):
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2, 2],
            strides=2,
        )  # out_size:30 x 80 x 32
                #with tf.name_scope("dropout1"):
        dropout1 = tf.layers.dropout(inputs=pool1, rate=dropout)

        # 第二层卷积
        conv2 = tf.layers.conv2d(
            inputs=dropout1,
            filters=64,
            kernel_size=[3, 3],
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )

        pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2, 2],
            strides=2
        )  # out_size:15 x 40 x 64

        dropout2 = tf.layers.dropout(inputs=pool2, rate=dropout)

        # 第三层卷积
        conv3 = tf.layers.conv2d(
             inputs=dropout2,
             filters=128,
             kernel_size=[3, 3],
             strides=1,
             padding='same',
             activation=tf.nn.relu
        )

        pool3 = tf.layers.max_pooling2d(
             inputs=conv3,
             pool_size=[2, 2],
             strides=2,
        ) # out_size 7 x 20 x 128

        dropout3 = tf.layers.dropout(inputs=pool3, rate=dropout)

        # 平坦化
        flat = tf.reshape(dropout3, [-1, 7 * 20 * 128])
        # 全连接层
        dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)

        dropout4 = tf.layers.dropout(dense, rate=dropout)
        # 预测的结果
        self.logits = tf.layers.dense(inputs=dropout4, units=MAX_CAPTCHA_LEN * CHAR_SET_LEN)

        # self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=self.logits)
        # self.loss = tf.losses.sigmoid_cross_entropy(self.y, self.logits)
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.logits))
            tf.summary.scalar("loss",self.loss)

        with tf.name_scope("train"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        # self.accuracy = tf.metrics.accuracy(labels=tf.argmax(self.y, 1),
        #                                     predictions=tf.argmax(self.logits, axis=1))[1]
        max_idx_p = tf.argmax(tf.reshape(self.logits, [-1, MAX_CAPTCHA_LEN, CHAR_SET_LEN]), 2)
        max_idx_l = tf.argmax(tf.reshape(self.y, [-1, MAX_CAPTCHA_LEN, CHAR_SET_LEN]), 2)
        with tf.name_scope("acc"):
            self.accuracy = tf.metrics.accuracy(labels=max_idx_l, predictions=max_idx_p)[1]
            tf.summary.scalar("acc1",self.accuracy)

            self.accuracy2 = tf.reduce_mean(
                tf.cast(tf.reduce_sum(tf.cast(tf.equal(max_idx_l, max_idx_p), tf.int32), axis=1) // MAX_CAPTCHA_LEN,
                    tf.float32))
            tf.summary.scalar("acc2",self.accuracy2)
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # with tf.Session() as sess:
        #     sess.run(init)
        #     if is_training == 1:
        #         Train(sess, input_x, output_y,MAX_CAPTCHA,CHAR_SET_LEN)
        #     else:
        #         Test(sess)

