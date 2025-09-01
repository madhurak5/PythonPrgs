# KDD Data
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

def get_train(*args):
    """Get training dataset for KDD 10 percent"""
    return _get_adapted_dataset("train")

def get_test(*args):
    """Get testing dataset for KDD 10 percent"""
    return _get_adapted_dataset("test")

def get_shape_input():
    """Get shape of the dataset for KDD 10 percent"""
    return (None, 121)

def get_shape_label():
    """Get shape of the labels in KDD 10 percent"""
    return (None,)

def _get_dataset():
    """ Gets the basic dataset
    Returns :
            dataset (dict): containing the data
                dataset['x_train'] (np.array): training images shape
                (?, 120)
                dataset['y_train'] (np.array): training labels shape
                (?,)
                dataset['x_test'] (np.array): testing images shape
                (?, 120)
                dataset['y_test'] (np.array): testing labels shape
                (?,)
    """
    col_names = _col_names()
    df = pd.read_csv("C://PythonPrgs/csvFiles/KDDTrain.csv", header=None, names=col_names)
    text_l = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']

    for name in text_l:
        _encode_text_dummy(df, name)

    labels = df['label'].copy()
    labels[labels != 'normal'] = 0
    labels[labels == 'normal'] = 1

    df['label'] = labels

    df_train = df.sample(frac=0.5, random_state=42)
    df_test = df.loc[~df.index.isin(df_train.index)]

    x_train, y_train = _to_xy(df_train, target='class')
    y_train = y_train.flatten().astype(int)
    x_test, y_test = _to_xy(df_test, target='class')
    y_test = y_test.flatten().astype(int)

    x_train = x_train[y_train != 1]
    y_train = y_train[y_train != 1]

    scaler = MinMaxScaler()
    scaler.fit(x_train)
    scaler.transform(x_train)
    scaler.transform(x_test)

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    return dataset

def _get_adapted_dataset(split):
    """ Gets the adapted dataset for the experiments
    Args :
            split (str): train or test
    Returns :
            (tuple): <training, testing> images and labels
    """
    dataset = _get_dataset()
    key_img = 'x_' + split
    key_lbl = 'y_' + split

    if split != 'train':
        dataset[key_img], dataset[key_lbl] = _adapt(dataset[key_img],
                                                    dataset[key_lbl])

    return (dataset[key_img], dataset[key_lbl])

def _encode_text_dummy(df, name):
    """Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1]
    for red,green,blue)
    """
    dummies = pd.get_dummies(df.loc[:,name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df.loc[:, dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

def _to_xy(df, target):
    """Converts a Pandas dataframe to the x,y inputs that TensorFlow needs"""
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    dummies = df[target]
    return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)

def _col_names():
    """Column names of the dataframe"""
    return ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

def _adapt(x, y, rho=0.2):
    """Adapt the ratio of normal/anomalous data"""

    # Normal data: label =0, anomalous data: label =1

    rng = np.random.RandomState(42) # seed shuffling

    inliersx = x[y == 0]
    inliersy = y[y == 0]
    outliersx = x[y == 1]
    outliersy = y[y == 1]

    size_outliers = outliersx.shape[0]
    inds = rng.permutation(size_outliers)
    outliersx, outliersy = outliersx[inds], outliersy[inds]

    size_test = inliersx.shape[0]
    out_size_test = int(size_test*rho/(1-rho))

    outestx = outliersx[:out_size_test]
    outesty = outliersy[:out_size_test]

    testx = np.concatenate((inliersx,outestx), axis=0)
    testy = np.concatenate((inliersy,outesty), axis=0)

    size_test = testx.shape[0]
    inds = rng.permutation(size_test)
    testx, testy = testx[inds], testy[inds]

    return testx, testy

# Run KDD
import time
import numpy as np
import tensorflow as tf
import logging
import importlib
import sys
# import gan.kdd_utilities as network
# import data.kdd as data
from sklearn.metrics import precision_recall_fscore_support


RANDOM_SEED =  146
FREQ_PRINT = 20 # print frequency image tensorboard [20]
STEPS_NUMBER = 500

def get_getter(ema):  # to update neural net with moving avg variables, suitable for ss learning cf Saliman
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var

    return ema_getter

def display_parameters(batch_size, starting_lr, ema_decay, weight, method, degree):
    '''See parameters
    '''
    print('Batch size: ', batch_size)
    print('Starting learning rate: ', starting_lr)
    print('EMA Decay: ', ema_decay)
    print('Weight: ', weight)
    print('Method for discriminator: ', method)
    print('Degree for L norms: ', degree)

def display_progression_epoch(j, id_max):
    '''See epoch progression
    '''
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush

def create_logdir(method, weight, rd):
    """ Directory to save training logs, weights, biases, etc."""
    return "gan/train_logs/kdd/{}/{}/{}".format(weight, method, rd)


def train_and_test(nb_epochs, weight, method, degree, random_seed):
    """ Runs the Bigan on the KDD dataset
    Note:
        Saves summaries on tensorboard. To display them, please use cmd line
        tensorboard --logdir=model.training_logdir() --port=number
    Args:
        nb_epochs (int): number of epochs
        weight (float, optional): weight for the anomaly score composition
        method (str, optional): 'fm' for ``Feature Matching`` or "cross-e"
                                     for ``cross entropy``, "efm" etc.
        anomalous_label (int): int in range 0 to 10, is the class/digit
                                which is considered outlier
    """
    logger = logging.getLogger("GAN.train.kdd.{}".format(method))

    # Placeholders
    input_pl = tf.placeholder(tf.float32, shape=data.get_shape_input(), name="input")
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    learning_rate = tf.placeholder(tf.float32, shape=(), name="lr_pl")

    # Data
    trainx, trainy = data.get_train()
    trainx_copy = trainx.copy()
    testx, testy = data.get_test()

    # Parameters
    starting_lr = network.learning_rate
    batch_size = network.batch_size
    latent_dim = network.latent_dim
    ema_decay = 0.9999

    rng = np.random.RandomState(RANDOM_SEED)
    nr_batches_train = int(trainx.shape[0] / batch_size)
    nr_batches_test = int(testx.shape[0] / batch_size)

    logger.info('Building training graph...')

    logger.warn("The GAN is training with the following parameters:")
    display_parameters(batch_size, starting_lr, ema_decay, weight, method, degree)

    gen = network.generator
    dis = network.discriminator

    # Sample noise from random normal distribution
    random_z = tf.random_normal([batch_size, latent_dim], mean=0.0, stddev=1.0, name='random_z')
    # Generate images with generator
    generator = gen(random_z, is_training=is_training_pl)
    # Pass real and fake images into discriminator separately
    real_d, inter_layer_real = dis(input_pl, is_training=is_training_pl)
    fake_d, inter_layer_fake = dis(generator, is_training=is_training_pl, reuse=True)

    with tf.name_scope('loss_functions'):
        # Calculate seperate losses for discriminator with real and fake images
        real_discriminator_loss = tf.losses.sigmoid_cross_entropy(tf.constant(1, shape=[batch_size]), real_d, scope='real_discriminator_loss')
        fake_discriminator_loss = tf.losses.sigmoid_cross_entropy(tf.constant(0, shape=[batch_size]), fake_d, scope='fake_discriminator_loss')
        # Add discriminator losses
        discriminator_loss = real_discriminator_loss + fake_discriminator_loss
        # Calculate loss for generator by flipping label on discriminator output
        generator_loss = tf.losses.sigmoid_cross_entropy(tf.constant(1, shape=[batch_size]), fake_d, scope='generator_loss')

    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        gvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        update_ops_gen = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
        update_ops_dis = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')

        optimizer_dis = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='dis_optimizer')
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='gen_optimizer')

        with tf.control_dependencies(update_ops_gen): # attached op for moving average batch norm
            gen_op = optimizer_gen.minimize(generator_loss, var_list=gvars)
        with tf.control_dependencies(update_ops_dis):
            dis_op = optimizer_dis.minimize(discriminator_loss, var_list=dvars)

        dis_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_dis = dis_ema.apply(dvars)

        with tf.control_dependencies([dis_op]):
            train_dis_op = tf.group(maintain_averages_op_dis)

        gen_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_gen = gen_ema.apply(gvars)

        with tf.control_dependencies([gen_op]):
            train_gen_op = tf.group(maintain_averages_op_gen)

    with tf.name_scope('training_summary'):
        with tf.name_scope('dis_summary'):
            tf.summary.scalar('real_discriminator_loss', real_discriminator_loss, ['dis'])
            tf.summary.scalar('fake_discriminator_loss', fake_discriminator_loss, ['dis'])
            tf.summary.scalar('discriminator_loss', discriminator_loss, ['dis'])

        with tf.name_scope('gen_summary'):
            tf.summary.scalar('loss_generator', generator_loss, ['gen'])


        sum_op_dis = tf.summary.merge_all('dis')
        sum_op_gen = tf.summary.merge_all('gen')

    logger.info('Building testing graph...')

    with tf.variable_scope("latent_variable"):
        z_optim = tf.get_variable(name='z_optim', shape= [batch_size, latent_dim], initializer=tf.truncated_normal_initializer())
        reinit_z = z_optim.initializer
    # EMA
    generator_ema = gen(z_optim, is_training=is_training_pl, getter=get_getter(gen_ema), reuse=True)
    # Pass real and fake images into discriminator separately
    real_d_ema, inter_layer_real_ema = dis(input_pl, is_training=is_training_pl, getter=get_getter(gen_ema), reuse=True)
    fake_d_ema, inter_layer_fake_ema = dis(generator_ema, is_training=is_training_pl, getter=get_getter(gen_ema), reuse=True)

    with tf.name_scope('error_loss'):
        delta = input_pl - generator_ema
        delta_flat = tf.contrib.layers.flatten(delta)
        gen_score = tf.norm(delta_flat, ord=degree, axis=1, keep_dims=False, name='epsilon')

    with tf.variable_scope('Discriminator_loss'):
        if method == "cross-e":
            dis_score = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(fake_d_ema), logits=fake_d_ema)

        elif method == "fm":
            fm = inter_layer_real_ema - inter_layer_fake_ema
            fm = tf.contrib.layers.flatten(fm)
            dis_score = tf.norm(fm, ord=degree, axis=1, keep_dims=False,
                             name='d_loss')

        dis_score = tf.squeeze(dis_score)

    with tf.variable_scope('Total_loss'):
        loss = (1 - weight) * gen_score + weight * dis_score

    with tf.variable_scope("Test_learning_rate"):
        step = tf.Variable(0, trainable=False)
        boundaries = [300, 400]
        values = [0.01, 0.001, 0.0005]
        learning_rate_invert = tf.train.piecewise_constant(step, boundaries, values)
        reinit_lr = tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope="Test_learning_rate"))

    with tf.name_scope('Test_optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate_invert).minimize(loss, global_step=step, var_list=[z_optim], name='optimizer')
        reinit_optim = tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope='Test_optimizer'))

    reinit_test_graph_op = [reinit_z, reinit_lr, reinit_optim]

    with tf.name_scope("Scores"):
        list_scores = loss

    logdir = create_logdir(method, weight, random_seed)

    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=None,
                             save_model_secs=120)

    logger.info('Start training...')
    with sv.managed_session() as sess:

        logger.info('Initialization done')

        writer = tf.summary.FileWriter(logdir, sess.graph)

        train_batch = 0
        epoch = 0

        while not sv.should_stop() and epoch < nb_epochs:

            lr = starting_lr

            begin = time.time()
            trainx = trainx[rng.permutation(trainx.shape[0])]  # shuffling unl dataset
            trainx_copy = trainx_copy[rng.permutation(trainx.shape[0])]

            train_loss_dis, train_loss_gen = [0, 0]
            # training
            for t in range(nr_batches_train):
                display_progression_epoch(t, nr_batches_train)

                # construct randomly permuted minibatches
                ran_from = t * batch_size
                ran_to = (t + 1) * batch_size

                # train discriminator
                feed_dict = {input_pl: trainx[ran_from:ran_to],
                             is_training_pl:True,
                             learning_rate:lr}
                _, ld, sm = sess.run([train_dis_op, discriminator_loss, sum_op_dis], feed_dict=feed_dict)
                train_loss_dis += ld
                writer.add_summary(sm, train_batch)

                # train generator
                feed_dict = {input_pl: trainx_copy[ran_from:ran_to],
                             is_training_pl:True,
                             learning_rate:lr}
                _, lg, sm = sess.run([train_gen_op, generator_loss, sum_op_gen], feed_dict=feed_dict)
                train_loss_gen += lg
                writer.add_summary(sm, train_batch)

                train_batch += 1

            train_loss_gen /= nr_batches_train
            train_loss_dis /= nr_batches_train

            logger.info('Epoch terminated')
            print("Epoch %d | time = %ds | loss gen = %.4f | loss dis = %.4f "
                  % (epoch, time.time() - begin, train_loss_gen, train_loss_dis))

            epoch += 1

        logger.warn('Testing evaluation...')
        inds = rng.permutation(testx.shape[0])
        testx = testx[inds]  # shuffling unl dataset
        testy = testy[inds]
        scores = []
        inference_time = []

        # Testing
        for t in range(nr_batches_test):

            # construct randomly permuted minibatches
            ran_from = t * batch_size
            ran_to = (t + 1) * batch_size
            begin_val_batch = time.time()

            # invert the gan
            feed_dict = {input_pl: testx[ran_from:ran_to],
                         is_training_pl:False}

            for step in range(STEPS_NUMBER):
                _ = sess.run(optimizer, feed_dict=feed_dict)
            scores += sess.run(list_scores, feed_dict=feed_dict).tolist()
            inference_time.append(time.time() - begin_val_batch)
            sess.run(reinit_test_graph_op)

        logger.info('Testing : mean inference time is %.4f' % (
            np.mean(inference_time)))
        ran_from = nr_batches_test * batch_size
        ran_to = (nr_batches_test + 1) * batch_size
        size = testx[ran_from:ran_to].shape[0]
        fill = np.ones([batch_size - size, 121])

        batch = np.concatenate([testx[ran_from:ran_to], fill], axis=0)
        feed_dict = {input_pl: batch,
                     is_training_pl: False}

        for step in range(STEPS_NUMBER):
            _ = sess.run(optimizer, feed_dict=feed_dict)
        batch_score = sess.run(list_scores,
                           feed_dict=feed_dict).tolist()

        scores += batch_score[:size]

        per = np.percentile(scores, 80)

        y_pred = scores.copy()
        y_pred = np.array(y_pred)

        inds = (y_pred < per)
        inds_comp = (y_pred >= per)

        y_pred[inds] = 0
        y_pred[inds_comp] = 1

        precision, recall, f1,_ = precision_recall_fscore_support(testy,
                                                                  y_pred,
                                                                  average='binary')
        print(
            "Testing : Prec = %.4f | Rec = %.4f | F1 = %.4f "
            % (precision, recall, f1))

def run(nb_epochs, weight, method, degree, label, random_seed=42):
    """ Runs the training process"""
    with tf.Graph().as_default():
        # Set the graph level seed
        tf.set_random_seed(random_seed)
        train_and_test(nb_epochs, weight, method, degree, random_seed)

# Adapt data
import numpy as np
import importlib

def adapt_labels(true_labels, label):
    """Adapt labels to anomaly detection context
    Args :
            true_labels (list): list of ints
            label (int): label which is considered anomalous
    Returns :
            true_labels (list): list of labels, 1 for anomalous and 0 for normal
    """
    if label == 0:
        (true_labels[true_labels == label], true_labels[true_labels != label]) = (0, 1)
        true_labels = [1] * true_labels.shape[0] - true_labels
    else:
        (true_labels[true_labels != label], true_labels[true_labels == label]) = (0, 1)

    return true_labels
# KDD Utilities File
import tensorflow as tf


"""Class for KDD10 percent GAN architecture.
Generator and discriminator.
"""

learning_rate = 0.00001
batch_size = 50
layer = 1
latent_dim = 32
dis_inter_layer_dim = 128
import  tensorflow as tf
# tf.initializers
# init_kernel = tf.contrib.layers.xavier_initializer()
init_kernel = tf.random_normal_initializer()

def generator(z_inp, is_training=False, getter=None, reuse=False):
    """ Generator architecture in tensorflow
    Generates data from the latent space
    Args:
        z_inp (tensor): variable in the latent space
        reuse (bool): sharing variables or not
    Returns:
        (tensor): last activation layer of the generator
    """
    with tf.variable_scope('generator', reuse=reuse, custom_getter=getter):

        name_net = 'layer_1'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(z_inp,
                                  units=64,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = tf.nn.relu(net, name='relu')

        name_net = 'layer_2'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=128,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = tf.nn.relu(net, name='relu')

        name_net = 'layer_4'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=121,
                                  kernel_initializer=init_kernel,
                                  name='fc')

        return net

def discriminator(x_inp, is_training=False, getter=None, reuse=False):
    """ Discriminator architecture in tensorflow
    Discriminates between real data and generated data
    Args:
        x_inp (tensor): input data for the encoder.
        reuse (bool): sharing variables or not
    Returns:
        logits (tensor): last activation layer of the discriminator (shape 1)
        intermediate_layer (tensor): intermediate layer for feature matching
    """
    with tf.variable_scope('discriminator', reuse=reuse, custom_getter=getter):

        name_net = 'layer_1'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(x_inp,
                                  units=256,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = leakyReLu(net)
            net = tf.layers.dropout(net, rate=0.2, name='dropout',
                                  training=is_training)

        name_net = 'layer_2'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=128,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = leakyReLu(net)
            net = tf.layers.dropout(net, rate=0.2, name='dropout',
                                  training=is_training)

        name_net = 'layer_3'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=dis_inter_layer_dim,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = leakyReLu(net)
            net = tf.layers.dropout(net,
                                    rate=0.2,
                                    name='dropout',
                                    training=is_training)

        intermediate_layer = net

        name_net = 'layer_4'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=1,
                                  kernel_initializer=init_kernel,
                                  name='fc')

        net = tf.squeeze(net)

        return net, intermediate_layer

def leakyReLu(x, alpha=0.1, name=None):
    if name:
        with tf.variable_scope(name):
            return _leakyReLu_impl(x, alpha)
    else:
        return _leakyReLu_impl(x, alpha)

def _leakyReLu_impl(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))
