# coding=utf-8
from utills import *
from network import *

dropout = 0.75
learning_rate = 0.001
print_stride = 50

training_steps = 5000
train_batch_size = 64
test_batch_size = 64


def train():
    """
    训练模型
    :return:
    """
    net = Model(dropout, learning_rate, MAX_CAPTCHA_LEN, CHAR_SET_LEN)
    saver = tf.train.Saver(max_to_keep=5)
    with tf.Session() as sess:
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_path, sess.graph)
        sess.run(net.init)

        for step in range(training_steps):
            batch_x, batch_y = gen_batch(train_batch_size)
            train_feeds = {net.x: batch_x, net.y: batch_y}
            train_loss, train_op = sess.run([net.loss, net.optimizer], train_feeds)
            if step % 10 == 0:
                print("第%d步，训练损失：%f" % (step, train_loss))
                rs = sess.run(merged_summary, feed_dict=train_feeds)
                writer.add_summary(rs,step)
            if step % print_stride == 0:
                batch_x, batch_y = gen_batch(test_batch_size)
                test_feeds = {net.x: batch_x, net.y: batch_y}
                test_acc = sess.run(net.accuracy, test_feeds)
                test_acc2 = sess.run(net.accuracy2, test_feeds)
                the_current_time()
                print("第%d步，训练损失：%f，测试准确率1：%f，测试准确率2: %f" % (step, train_loss, test_acc, test_acc2))
                if test_acc > 0.5 and test_acc2 > 0.5:
                    saver.save(sess, "./model/Captcha_Identification.model", global_step=step)
                    break


if __name__ == '__main__':
    train()
