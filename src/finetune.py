import os
import numpy as np
import tensorflow as tf

from src.alexnet import AlexNet
from src.datagenerator import ImageDataGenerator
from datetime import datetime

from tensorflow.contrib.data import Iterator

train_file = '/path/to/train.txt'
val_file = '/path/to/val.txt'

learning_rate = 0.01
num_epochs = 10
batch_size = 128


dropout_rate = 0.5
num_classes = 2
train_layers = ['fc8', 'fc7', 'fc6']
display_step = 20
filewriter_path = '/tmp/finetune_alexnet/tensorboard'
checkpoint_path = '/tmp/finetune_alexnet/checkpoints'


if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file, mode='training', batch_size=batch_size,
                                 num_classes=num_classes, shuffle=True)
    val_data = ImageDataGenerator(train_file, mode='inference', batch_size=batch_size,
                                  num_classes=num_classes, shuffle=False)
    iterator = Iterator.from_structure(tr_data.data.output_types, tr_data.data.output_shapes)
    next_batch = iterator.get_next()

training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

x = tf.placeholder(tf.float32, [batch_size, 277, 277, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])

keep_prob = tf.placeholder(tf.float32)

model = AlexNet(x, keep_prob, num_classes, train_layers)

score = model.fc8

var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

with tf.name_scope('cross_ent'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))

with tf.name_scope('train'):
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

for gradients, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradients)

for var in var_list:
    tf.summary.histogram(var.name, var)

with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.arg_max(score, 1), tf.arg_max((y, 1)))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar('accuracy', accuracy)
merged_summary = tf.summary.merge_all()

writer = tf.summary.FileWriter(filewriter_path)


saver = tf.train.Saver()

train_batches_per_epoch = int(np.floor(tr_data.data_size / batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)

    model.lodel_initial_weights(sess)

    for epoch in range(num_epochs):
        sess.run(training_init_op)

        for step in range(train_batches_per_epoch):
            img_batch, label_batch = sess.run(next_batch)

            sess.run(train_op, feed_dict={x: img_batch, y: label_batch, keep_prob: dropout_rate})

            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: img_batch, y:label_batch, keep_prob: 1})

                writer.add_summary(s, epoch * train_batches_per_epoch +step)
        sess.run(validation_init_op)
        test_acc = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):
            img_batch, label_batch = sess.run(next_batch)
            acc = sess.run(accuracy, feed_dict={x: img_batch, y: label_batch, keep_prob: 1})

            test_acc += acc
            test_count += 1
        test_acc /= test_count

        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch{}.ckpt'.format(epoch + 1))

        save_path = saver.save(sess, checkpoint_name)

