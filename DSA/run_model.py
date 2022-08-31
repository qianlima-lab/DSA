# -*- coding: utf-8 -*-

import tensorflow as tf
from model.dsa import model as dsa_model


import numpy as np
import math
from model.dsa import load_data
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("log_file_name", "dsa_test", "")
flags.DEFINE_string("hlf_win_len", "[1,5,10,15]", "learn win model parameter.")

flags.DEFINE_string("gpu", "0", "which gpu to use.")
flags.DEFINE_integer("epochs", 100, "Batch size to use during training.")
flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
flags.DEFINE_integer("hidden_dim", 150, "Batch size to use during training.")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
flags.DEFINE_float("x_keep_prob", 0.5, "Learning rate.")
flags.DEFINE_float("drop_keep_prob", 0.5, "Learning rate.")
flags.DEFINE_integer("num_class", 5, 'num_class.')

# os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
handler = logging.FileHandler(FLAGS.log_file_name + ".txt",mode='a', encoding='utf-8')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)

logger.addHandler(handler)
logger.addHandler(console)

logger.info(">>>>> Start print log <<<<<")

def train(sess,m,batch_nums,data_generate,data):
    losses = np.zeros(data.shape[0])
    accs = np.zeros(data.shape[0])

    cnt = 0
    for step_i in range(batch_nums):
        input_x, input_y, input_len = data_generate.next()
        max_len = input_x.shape[1]
        input_len_new = []
        for input_leni in input_len:
            if input_leni > max_len:
                input_len_new.append(max_len)
            else:
                input_len_new.append(input_leni)

        feed_dict = {m.x: input_x, m.y: input_y, m.sequence_length: input_len_new,
                     m.drop_keep_prob: FLAGS.x_keep_prob, m.x_keep_prob: FLAGS.x_keep_prob, m.this_batch_size: len(input_len),
                     m.is_train: True}

        _, correct_predict, cross_entropy = sess.run(
            [m.train_step, m.correct_predict, m.cross_entropy], feed_dict)

        srt = step_i * batch_size
        end = (step_i + 1) * batch_size if step_i != batch_nums - 1 else data.shape[0]

        losses[srt:end] = cross_entropy
        accs[srt:end] = correct_predict
        cnt += correct_predict.shape[0]

    assert cnt == data.shape[0]
    loss = np.mean(losses)
    acc = np.mean(accs)
    return loss, acc

def dev_test(sess,m,batch_nums,data_generate,data):
    losses = np.zeros(data.shape[0])
    accs = np.zeros(data.shape[0])

    cnt = 0
    for step_i in range(batch_nums):

        input_x, input_y, input_len = data_generate.next()

        max_len = input_x.shape[1]
        input_len_new = []
        for input_leni in input_len:
            if input_leni > max_len:
                input_len_new.append(max_len)
            else:
                input_len_new.append(input_leni)

        feed_dict = {m.x: input_x, m.y: input_y, m.sequence_length: input_len_new,
                     m.drop_keep_prob: 1.0, m.x_keep_prob: 1.0, m.this_batch_size: len(input_len),
                     m.is_train: False}
        correct_predict, cross_entropy = sess.run(
            [m.correct_predict, m.cross_entropy], feed_dict)

        srt = step_i * batch_size
        end = (step_i + 1) * batch_size if step_i != batch_nums - 1 else data.shape[0]

        losses[srt:end] = cross_entropy
        accs[srt:end] = correct_predict
        cnt += correct_predict.shape[0]

    assert cnt == data.shape[0]
    loss = np.mean(losses)
    acc = np.mean(accs)
    return loss, acc


def one_flod():

    graph = tf.Graph()
    with graph.as_default():
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        m = dsa_model(FLAGS, parameter_configs)

        logger.info( "parameter space: " + str(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())

            data_generate_train = load_data(train_data, train_labels, train_length, batch_size)
            data_generate_dev = load_data(dev_data, dev_labels, dev_length, batch_size)
            data_generate_test = load_data(test_data, test_labels, test_length, batch_size)
            train_batch_nums = int(math.ceil(train_data.shape[0] / (batch_size + 0.0)))
            dev_batch_nums = int(math.ceil(dev_data.shape[0] / (batch_size + 0.0)))
            test_batch_nums = int(math.ceil(test_data.shape[0] / (batch_size + 0.0)))

            logger.info("train_batch_num: %d, %d" % (train_batch_nums, test_batch_nums))
            logger.info('length :%d'%train_data.shape[1])


            max_dev_accuracy = 0
            max_dev_test_accuracy = 0

            for epoch_i in range(FLAGS.epochs):
                train_loss, train_acc = train(sess,m,train_batch_nums,data_generate_train, train_data)
                logger.info("epochs: %d, train acc: %f, loss: %f" % (epoch_i, train_acc, train_loss))
                dev_loss, dev_acc = dev_test(sess, m, dev_batch_nums, data_generate_dev, dev_data)
                test_loss, test_acc = dev_test(sess, m, test_batch_nums, data_generate_test, test_data)


                if dev_acc > max_dev_accuracy:
                    max_dev_accuracy = dev_acc
                    max_dev_test_accuracy = test_acc

                    logger.info("max dev acc: %f, dev loss: %f" % (dev_acc, dev_loss))
                    logger.info("test acc: %f, test loss: %f" % (test_acc, test_loss))


        logger.info("max valid: {}, max_test_acc:{}".format(max_dev_accuracy,max_dev_test_accuracy))
        logger.info('----------------- outputs  down ------------------')
        # print '----------------- outputs  down ------------------'
    return max_dev_accuracy,max_dev_test_accuracy


if __name__ == '__main__':

    dataset_name  = 'dataset/test_sample/'
    """
    Prepare datasets below.
    train_data/dev_data/test_data: list of sentences, each sentence is processed to a list of word ids.
    train_labels/dev_labels/test_labels: list of label for each sentence.
    train_length/dev_length/test_length: list of length for each sentence.
    """
    embedding = np.load(dataset_name  +'test_sample.glove.npz')['embeddings']

    train_data = np.load(dataset_name + 'train.sen.ids.sample.npz')['sentences']
    train_labels = np.load(dataset_name + 'train.label.ids.sample.npz')['labels']
    train_length = np.load(dataset_name + 'train.len.ids.sample.npz')['lengths']

    dev_data = np.load(dataset_name + 'valid.sen.ids.sample.npz')['sentences']
    dev_labels = np.load(dataset_name + 'valid.label.ids.sample.npz')['labels']
    dev_length = np.load(dataset_name + 'valid.len.ids.sample.npz')['lengths']

    test_data = np.load(dataset_name + 'test.sen.ids.sample.npz')['sentences']
    test_labels = np.load(dataset_name + 'test.label.ids.sample.npz')['labels']
    test_length = np.load(dataset_name + 'test.len.ids.sample.npz')['lengths']

    batch_size = FLAGS.batch_size
    parameter_configs = {
                         "n_vocab" : embedding.shape[0],
                         "vocab_size":embedding.shape[0],
                         "embedding": embedding,
                         "n_steps":train_data.shape[1],
                         "num_class": FLAGS.num_class,
                          }

    logger.info('Applying Parameters:')
    for k in FLAGS:
        logger.info('%s: %s' % (k, str(FLAGS[k].value)))

    for p in parameter_configs:
        if p == "embedding": continue
        logger.info('%s: %s' % (p, str(parameter_configs[p])))

    dev_accs = []
    dev_test_accs = []

    for run_i in range(3):
        a_dev_accuracy, a_dev_test_accuracy = one_flod()
        dev_accs.append(a_dev_accuracy)
        dev_test_accs.append(a_dev_test_accuracy)

    logger.info('-' * 40)
    for run_i,(a,b) in enumerate(zip(dev_accs,dev_test_accs)):
        logger.info('run_{}: max dev acc: {}, test acc: {}'.format(run_i,a,b))
    logger.info('-' * 40)