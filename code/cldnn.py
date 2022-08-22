# -*- coding: utf-8 -*-
"""
Created on Thu march  3 15:44:27 2022

@author: Focuslab_LK
"""
from keras.models import load_model
import h5py
import pandas as pd
import numpy as np
from numpy import array
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers.core import Reshape, Dropout, Activation
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers.convolutional import Conv1D, ZeroPadding1D
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import keras
import os
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from matplotlib import pyplot as plt
import itertools

tf.compat.v1.reset_default_graph()
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


def loadtestdata(i):
    testFile = f'data/Multi_task_{i}dB.mat'  # 文件名字
    data_test = h5py.File(testFile, 'r')
    x = data_test['data'][:]
    y = data_test['RFF_label'][:]  # y的数据格式为1*6200
    x = x.transpose(2, 1, 0)
    y = y.transpose(1, 0)  # xyz 变为yx 变为6200*1
    y = to_categorical(y)
    return x, y


def plot_confuse(model, x_val, y_val):
    predictions = model.predict(x_val, batch_size=16)
    # predictions = model.predict(img)
    predictions = np.argmax(predictions, axis=1)

    truelabel = y_val.argmax(axis=-1)  # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    print(conf_mat)
    # plt.figure()
    np.savetxt((file_name + 'Confusion_matrix.txt'), conf_mat, fmt='%d', delimiter=',')


def save_loss_acc(history1, file_name):
    accy = history1.history['accuracy']
    lossy = history1.history['loss']
    np_accy = np.array(accy).reshape((1, len(accy)))  # reshape是为了能够跟别的信息组成矩阵一起存储
    np_lossy = np.array(lossy).reshape((1, len(lossy)))
    np_out = np.concatenate([np_accy, np_lossy], axis=0)
    file_name1 = file_name + f'loss_acc.txt'
    np.savetxt(file_name1, np_out)

    val_accy = history1.history['val_accuracy']
    val_lossy = history1.history['val_loss']
    np_val_accy = np.array(val_accy).reshape((1, len(val_accy)))  # reshape是为了能够跟别的信息组成矩阵一起存储
    np_val_lossy = np.array(val_lossy).reshape((1, len(val_lossy)))
    np_val_out = np.concatenate([np_val_accy, np_val_lossy], axis=0)
    file_name2 = file_name + 'val_loss_acc.txt'
    np.savetxt(file_name2, np_val_out)


def test_model(model):
    [Loss, Acc] = model.evaluate(X_test, Y_test, batch_size=10, verbose=0)
    result_acc_loss = [Loss, Acc]
    result_acc_loss = array(result_acc_loss)
    result_acc_loss = result_acc_loss.reshape(2, 1).T
    df = pd.DataFrame(result_acc_loss, columns=['Test Loss', 'Test Acc'])
    # df.to_excel("device4/cnn_result_acc_loss_snr=" + str(snr) + ".xlsx", index=False)
    df.to_excel(file_name + f"test_loss_acc.xlsx", index=False)


for i in [0, 5, 10, 15, 20]:
    for a in [1, 2]:
        fontsize = 35

        s = 6
        labels = ['PA1', "PA2", "PA3", 'PA4', 'PA5', "PA6"]
        dr = 0.5
        beach_size = 32
        epoch = 100
        file_name = f'result/{a}/'

        x, y = loadtestdata(i)
        X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size=0.2, random_state=30)
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.125, random_state=30)

        in_shp = Input(shape=(6000, 2))

        input_shp = ZeroPadding1D((2))(in_shp)
        layer11 = Conv1D(128, 10, padding='valid', activation="relu", name="conv11",
                         kernel_initializer='glorot_uniform',
                         data_format="channels_last")(input_shp)
        layer11 = Dropout(dr)(layer11)
        layer11_padding = ZeroPadding1D((2))(layer11)
        layer12 = Conv1D(128, 10, padding="valid", activation="relu", name="conv12",
                         kernel_initializer='glorot_uniform',
                         data_format="channels_last")(layer11_padding)
        layer12 = Dropout(dr)(layer12)
        layer12_padding = ZeroPadding1D((2))(layer12)
        layer13 = Conv1D(128, 10, padding='valid', activation="relu", name="conv13",
                         kernel_initializer='glorot_uniform',
                         data_format="channels_last")(layer12_padding)
        layer13 = Dropout(dr)(layer13)

        concat = keras.layers.concatenate([layer11, layer13], axis=1)
        concat_size = list(np.shape(concat))
        input_dim = int(concat_size[-2])
        timesteps = int(concat_size[-1])
        concat = Reshape((timesteps, input_dim))(concat)
        lstm_out = LSTM(64, input_dim=input_dim, input_length=timesteps)(concat)  # 在

        layer_dense1 = Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1")(lstm_out)
        layer_dropout1 = Dropout(dr)(layer_dense1)
        layer_dense2 = Dense(128, activation='relu', kernel_initializer='he_normal', name="dense2")(layer_dropout1)
        layer_dropout2 = Dropout(dr)(layer_dense2)
        layer_dense3 = Dense(s, kernel_initializer='he_normal', name="dense3")(layer_dropout2)
        layer_softmax = Activation('softmax')(layer_dense3)
        output = Reshape([s])(layer_softmax)

        model = Model(inputs=in_shp, outputs=output)

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        checkpoint = ModelCheckpoint(filepath=file_name + f"every_epoch_model.hdf5", verbose=1, save_best_only=True)

        tensorboard = TensorBoard(log_dir=file_name + f"picture.log",
                                  histogram_freq=0, )
        earlystopping = EarlyStopping(monitor="val_loss", patience=40, verbose=1, mode="auto")
        reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1,
                                     mode='auto')
        model.summary()

        history1 = model.fit(
            X_train,
            Y_train,
            batch_size=beach_size,
            epochs=epoch,
            verbose=1,
            validation_data=(X_val, Y_val),
            callbacks=[checkpoint, tensorboard, reducelr]
        )

        save_loss_acc(history1, file_name)
        test_model(model)
        model.save(file_name + f'model.h5')
        plot_confuse(model, X_test, Y_test)
