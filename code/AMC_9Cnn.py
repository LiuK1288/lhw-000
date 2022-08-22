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
from keras.layers.core import Reshape, Dropout, Activation
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers.convolutional import Conv1D

from keras.models import Model
from keras.layers import Input, Dense, concatenate, MaxPooling1D, Flatten, BatchNormalization
import os
from sklearn.metrics import confusion_matrix
import tensorflow as tf

# from Network import RealCNN

tf.compat.v1.reset_default_graph()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


# 导入数据函数
def loadtestdata(SNR):
    testFile = f'data/Multi_data{SNR}dB.mat'
    data_test = h5py.File(testFile, 'r')
    x = data_test['data'][:]
    y = data_test['AMC_label'][:]
    x = x.transpose(2, 1, 0)
    y = y.transpose(1, 0)
    y = to_categorical(y)
    return x, y


def RealCNN():
    x_input = Input(shape=(6000, 2))
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
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x_output = Dense(classs, activation='softmax')(x)
    model = Model(inputs=x_input,
                  outputs=x_output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model


def plot_confuse(model, x_val, y_val):
    predictions = model.predict(x_val, batch_size=batch_size)
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
    np_out = concatenate([np_accy, np_lossy], axis=0)
    file_name1 = file_name + f'loss_acc.txt'
    np.savetxt(file_name1, np_out)

    val_accy = history1.history['val_accuracy']
    val_lossy = history1.history['val_loss']
    np_val_accy = np.array(val_accy).reshape((1, len(val_accy)))  # reshape是为了能够跟别的信息组成矩阵一起存储
    np_val_lossy = np.array(val_lossy).reshape((1, len(val_lossy)))
    np_val_out = concatenate([np_val_accy, np_val_lossy], axis=0)
    file_name2 = file_name + 'val_loss_acc.txt'
    np.savetxt(file_name2, np_val_out)


def test_model(model):
    [Loss, Acc] = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=0)
    print(f'snr = {SNR} {a},acc = {Acc}')
    result_acc_loss = [Loss, Acc]
    result_acc_loss = array(result_acc_loss)
    result_acc_loss = result_acc_loss.reshape(2, 1).T
    df = pd.DataFrame(result_acc_loss, columns=['Test Loss', 'Test Acc'])
    # 保存到本地excel
    df.to_excel(file_name + f"test_loss_acc.xlsx", index=False)


def Data_Split(x, y, V, T):
    choose_index = np.load('result/choose_index.npy')
    train_index = choose_index[0:int(y.shape[0] * (1 - V - T))]
    val_index = choose_index[int(y.shape[0] * (1 - V - T)):int(y.shape[0] * (1 - T))]
    test_index = choose_index[int(y.shape[0] * (1 - T)):]
    return x[train_index], y[train_index], x[val_index], y[val_index], x[test_index], y[test_index]


classs = 8
dr = 0.2
batch_size = 32
epoch = 100

for SNR in [0, 5, 10, 15, 20]:
    # 导入数据
    x, y = loadtestdata(SNR)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = Data_Split(x, y, 0.2, 0.1)  # 7:2:1

    for a in [1, 2]:
        file_name = f'result/WY/AMC/{SNR}dB/AMC{a}/'

        model = RealCNN()

        checkpoint = ModelCheckpoint(filepath=file_name + f"every_epoch_model.hdf5", verbose=1, save_best_only=True)

        tensorboard = TensorBoard(log_dir=file_name + f"picture.log",
                                  histogram_freq=0, )
        earlystopping = EarlyStopping(monitor="val_loss", patience=32, verbose=1, mode="auto")
        reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=1,
                                     mode='auto')

        history1 = model.fit(
            X_train,
            Y_train,
            batch_size=batch_size,
            epochs=epoch,
            verbose=1,
            validation_data=(X_val, Y_val),
            callbacks=[checkpoint, tensorboard, reducelr, earlystopping]
        )

        save_loss_acc(history1, file_name)
        test_model(model)
        model.save(file_name + f'model.h5')
        plot_confuse(model, X_test, Y_test)
