# -*- coding: utf-8 -*-
"""
Created on Thu march  3 15:44:27 2022

@author: Focuslab_LK
"""

import h5py
import pandas as pd
import numpy as np
from numpy import array
from keras.utils import to_categorical
from keras.layers.core import Reshape, Dropout, Activation, Flatten
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers.convolutional import Conv1D
from keras.models import Model
from keras.layers import concatenate, Input, Dense, BatchNormalization, MaxPooling1D
import os
from sklearn.metrics import confusion_matrix
import tensorflow as tf

tf.compat.v1.reset_default_graph()
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


def loadtestdata(snr):
    testFile = f'data/Multi_data{snr}dB.mat'  # 文件名字
    data_test = h5py.File(testFile, 'r')
    x = data_test['data'][:]
    y = data_test['RFF_label'][:]
    z = data_test['AMC_label'][:]

    x = x.transpose(2, 1, 0)
    y = y.transpose(1, 0)
    z = z.transpose(1, 0)

    y = to_categorical(y)
    z = to_categorical(z)

    return x, y, z


def plot_confuse(x_test, y_test, z_test):
    predictions1, predictions2 = model.predict(x_test, batch_size=batch_size)
    predictions1 = np.argmax(predictions1, axis=1)
    predictions2 = np.argmax(predictions2, axis=1)

    truelabel1 = y_test.argmax(axis=-1)
    truelabel2 = z_test.argmax(axis=-1)

    conf_mat1 = confusion_matrix(y_true=truelabel1, y_pred=predictions1)
    conf_mat2 = confusion_matrix(y_true=truelabel2, y_pred=predictions2)
    print('*-*' * 20)
    print('*-*' * 20)
    print('RFF')
    print(conf_mat1)
    print('*-*' * 20)
    print('AMC')
    print(conf_mat2)
    print('*-*' * 20)
    print('*-*' * 20)
    np.savetxt((file_name + 'RFF_Confusion_matrix.txt'), conf_mat1, fmt='%d', delimiter=',')
    np.savetxt((file_name + 'AMC_Confusion_matrix.txt'), conf_mat2, fmt='%d', delimiter=',')


def save_loss_acc():
    accy = history1.history['output1_accuracy']
    lossy = history1.history['output1_loss']
    np_accy = np.array(accy).reshape((1, len(accy)))
    np_lossy = np.array(lossy).reshape((1, len(lossy)))
    np_out = concatenate([np_accy, np_lossy], axis=0)
    file_name1 = file_name + f'RFF_loss_acc.txt'
    np.savetxt(file_name1, np_out)

    val_accy = history1.history['val_output1_accuracy']
    val_lossy = history1.history['val_output1_loss']
    np_val_accy = np.array(val_accy).reshape((1, len(val_accy)))
    np_val_lossy = np.array(val_lossy).reshape((1, len(val_lossy)))
    np_val_out = concatenate([np_val_accy, np_val_lossy], axis=0)
    file_name2 = file_name + 'RFF_val_loss_acc.txt'
    np.savetxt(file_name2, np_val_out)

    accy = history1.history['output2_accuracy']
    lossy = history1.history['output2_loss']
    np_accy = np.array(accy).reshape((1, len(accy)))
    np_lossy = np.array(lossy).reshape((1, len(lossy)))
    np_out = concatenate([np_accy, np_lossy], axis=0)
    file_name1 = file_name + f'AMC_loss_acc.txt'
    np.savetxt(file_name1, np_out)

    val_accy = history1.history['val_output2_accuracy']
    val_lossy = history1.history['val_output2_loss']
    np_val_accy = np.array(val_accy).reshape((1, len(val_accy)))
    np_val_lossy = np.array(val_lossy).reshape((1, len(val_lossy)))
    np_val_out = concatenate([np_val_accy, np_val_lossy], axis=0)
    file_name2 = file_name + 'AMC_val_loss_acc.txt'
    np.savetxt(file_name2, np_val_out)

    train_loss = history1.history['loss']
    val_loss = history1.history['val_loss']
    train_loss = np.array(train_loss).reshape((1, len(train_loss)))
    val_loss = np.array(val_loss).reshape((1, len(val_loss)))
    loss_out = concatenate([train_loss, val_loss], axis=0)
    file_name2 = file_name + 'train_loss.txt'
    np.savetxt(file_name2, loss_out)


def test_model():
    [loss, Loss1, Loss2, Acc1, Acc2] = model.evaluate(X_test, [Y_test, Z_test], batch_size=batch_size, verbose=0)
    result_acc_loss = [loss, Loss1, Acc1, Loss2, Acc2]
    result_acc_loss = array(result_acc_loss)
    result_acc_loss = result_acc_loss.reshape(5, 1).T
    df = pd.DataFrame(result_acc_loss,
                      columns=['Test loss', 'Test Loss1', 'Test Acc_RFF', 'Test Loss2', 'Test ACC_AMC'])
    df.to_excel(file_name + f"test_loss_acc.xlsx", index=False)


def Data_Split(x, y, z, V, T):
    choose_index = np.load('result/choose_index.npy')
    train_index = choose_index[0:int(y.shape[0] * (1 - V - T))]
    val_index = choose_index[int(y.shape[0] * (1 - V - T)):int(y.shape[0] * (1 - T))]
    test_index = choose_index[int(y.shape[0] * (1 - T)):]
    return x[train_index], y[train_index], z[train_index], x[val_index], y[val_index], z[val_index], x[test_index], y[
        test_index], z[test_index]


def RealCNN():
    x_input = Input(shape=(6000, 2), name='x_input')
    x = Conv1D(128, 3, activation='relu', padding='same')(x_input)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)

    x1 = Dense(512, activation='relu', name='dense1')(x)
    x1 = Dropout(0.5)(x1)
    output1 = Dense(RFF_s, activation='softmax', name='output1')(x1)

    x2 = Dense(512, activation='relu', name='dense2')(x)
    x2 = Dropout(0.5)(x2)
    output2 = Dense(AMC_s, activation='softmax', name='output2')(x2)

    model = Model(inputs=x_input, outputs=[output1, output2])
    model.compile(optimizer='adam',
                  loss={'output1': 'categorical_crossentropy',
                        'output2': 'categorical_crossentropy'},
                  loss_weights={'output1': 0.8,
                                'output2': 0.2},
                  metrics=['accuracy'])
    model.summary()
    return model


RFF_s = 6
AMC_s = 8
batch_size = 32
epoch = 100
for snr in [0, 5, 10, 15, 20]:
    x, y, z = loadtestdata(snr)
    X_train, Y_train, Z_train, X_val, Y_val, Z_val, X_test, Y_test, Z_test = Data_Split(x, y, z, 0.2, 0.1)  # 7:2:1
    for a in [1, 2]:
        file_name = f'result/WY/RFF+AMC/{snr}dB/{a}/'
        model = RealCNN()

        checkpoint = ModelCheckpoint(filepath=file_name + f"every_epoch_model.hdf5", verbose=1, save_best_only=True)

        tensorboard = TensorBoard(log_dir=file_name + f"picture.log",
                                  histogram_freq=0, )
        earlystopping = EarlyStopping(monitor="val_output1_loss", patience=32, verbose=1, mode="auto")

        reducelr = ReduceLROnPlateau(monitor='val_output1_loss', factor=0.5, patience=8, verbose=1, mode='auto')

        history1 = model.fit({'x_input': X_train, },
                             {'output1': Y_train,
                              'output2': Z_train},
                             epochs=epoch,
                             batch_size=batch_size,
                             verbose=1,
                             validation_data=([X_val], [Y_val, Z_val]),
                             callbacks=[checkpoint, tensorboard, reducelr, earlystopping]
                             )

        model.save(file_name + f'model.h5')
        test_model()
        plot_confuse(X_test, Y_test, Z_test)
        save_loss_acc()
