#coding=gbk


#�������ݷ������ÿ�

import pandas as pd

import numpy as np

import tensorflow as tf

from sklearn.metrics import mean_absolute_error,mean_squared_error

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')




filename = r'.\61111914003-lstm.csv'
dt = pd.read_csv(filename)
data= dt.values




#���峣��

rnn_unit=10       #hidden layer units

input_size=4

output_size=1

lr=0.0006         #ѧϰ��

tf.reset_default_graph()

#����㡢�����Ȩ�ء�ƫ��

weights={

         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),

         'out':tf.Variable(tf.random_normal([rnn_unit,1]))

         }

biases={

        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),

        'out':tf.Variable(tf.constant(0.1,shape=[1,]))

        }





def get_data(batch_size=60, time_step=20, train_begin=0, train_end=18290):
    batch_index = []

    scaler_for_x = MinMaxScaler(feature_range=(0, 1))  # ������minmax����

    scaler_for_y = MinMaxScaler(feature_range=(0, 1))

    scaled_x_data = scaler_for_x.fit_transform(data[:, :-1])

    scaled_y_data = scaler_for_y.fit_transform(data[:, -1,np.newaxis])

    label_train = scaled_y_data[train_begin:train_end]

    label_test = scaled_y_data[train_end:]

    normalized_train_data = scaled_x_data[train_begin:train_end]

    normalized_test_data = scaled_x_data[train_end:]

    train_x, train_y = [], []  # ѵ����x��y������

    for i in range(len(normalized_train_data) - time_step):

        if i % batch_size == 0:
            batch_index.append(i)

        x = normalized_train_data[i:i + time_step, :4]

        y = label_train[i:i + time_step, np.newaxis]

        train_x.append(x.tolist())

        train_y.append(y.tolist())

    batch_index.append((len(normalized_train_data) - time_step))

    size = (len(normalized_test_data) + time_step - 1) // time_step  # ��size��sample

    test_x, test_y = [], []

    for i in range(size - 1):
        x = normalized_test_data[i * time_step:(i + 1) * time_step, :4]

        y = label_test[i * time_step:(i + 1) * time_step]

        test_x.append(x.tolist())

        test_y.extend(y)

    test_x.append((normalized_test_data[(i + 1) * time_step:, :4]).tolist())

    test_y.extend((label_test[(i + 1) * time_step:]).tolist())

    train_y = np.reshape(train_y,[-1,time_step,1])

    return batch_index, train_x, train_y, test_x, test_y, scaler_for_y




#�������������������������������������������������������������������������������������

def lstm(X):

    batch_size=tf.shape(X)[0]

    time_step=tf.shape(X)[1]

    w_in=weights['in']

    b_in=biases['in']

    input=tf.reshape(X,[-1,input_size])  #��Ҫ��tensorת��2ά���м��㣬�����Ľ����Ϊ���ز������

    input_rnn=tf.matmul(input,w_in)+b_in

    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #��tensorת��3ά����Ϊlstm cell������

    # cell=tf.contrib.rnn.BasicLSTMCell(rnn_unit)
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)

    #cell=tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(rnn_unit)

    init_state=cell.zero_state(batch_size,dtype=tf.float32)

    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn�Ǽ�¼lstmÿ������ڵ�Ľ����final_states�����һ��cell�Ľ��

    output=tf.reshape(output_rnn,[-1,rnn_unit]) #��Ϊ����������

    w_out=weights['out']

    b_out=biases['out']

    pred=tf.matmul(output,w_out)+b_out

    return pred,final_states


# ������������������������������������ѵ��ģ�͡�����������������������������������

def train_lstm(batch_size=80, time_step=15, train_begin=0, train_end=18290):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])

    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])

    batch_index, train_x, train_y, test_x, test_y, scaler_for_y = get_data(batch_size, time_step, train_begin,
                                                                           train_end)

    pred, _ = lstm(X)

    # ��ʧ����

    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))

    train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    saver=tf.train.Saver(tf.global_variables(),max_to_keep=5)
    module_file = tf.train.latest_checkpoint('sm/')

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        if module_file != None:
            saver.restore(sess, module_file)

        # �ظ�ѵ��5000��

        iter_time = 200

        for i in range(iter_time):

            for step in range(len(batch_index) - 1):

                a = train_x[batch_index[step]:batch_index[step + 1]]

                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                                 Y: train_y[batch_index[step]:batch_index[step + 1]]})

            if i % 200 == 0:
                print('iter:', i, 'loss:', loss_)

            if i % 200==0:
                print("����ģ�ͣ�",saver.save(sess,'sm/stock2.model',global_step=i))

        ####predict####

    #     test_predict = []
    #
    #     for step in range(len(test_x)):
    #         prob = sess.run(pred, feed_dict={X: [test_x[step]]})
    #
    #         predict = prob.reshape((-1))
    #
    #         test_predict.extend(predict)
    #
    #     test_predict = scaler_for_y.inverse_transform(test_predict)
    #
    #     test_y = scaler_for_y.inverse_transform(test_y)
    #
    #     rmse = np.sqrt(mean_squared_error(test_predict, test_y))
    #
    #     mae = mean_absolute_error(y_pred=test_predict, y_true=test_y)
    #
    #     print('mae:', mae, '   rmse:', rmse)
    #
    # return test_predict



def predict(batch_size=80, time_step=10, train_begin=0, train_end=18290):

    test_predict = []
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])

    batch_index, train_x, train_y, test_x, test_y, scaler_for_y = get_data(batch_size, time_step, train_begin,
                                                                           train_end)
    pred, _ = lstm(X)

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables())
        module_file = tf.train.latest_checkpoint('sm-1/')
        saver.restore(sess, module_file)


        # for step in range(len(test_x)):
        #     prob = sess.run(pred, feed_dict={X: [test_x[step]]})
        #
        #     predict = prob.reshape((-1))
        #
        #     test_predict.extend(predict)
        #     print(step)



        prob = sess.run(pred, feed_dict={X: [test_x[0]]})

        # test_d = np.ravel(test_x,'C')
        #
        # prob = sess.run(pred, feed_dict={X: [test_d]})

        predict = prob.reshape((-1))

        test_predict.extend(predict)


        test_predict = scaler_for_y.inverse_transform(test_predict)

        plt.figure(figsize=(24, 8))

        plt.plot(data[train_end:train_end+time_step,-1],color='b')
        plt.plot(test_predict,color='r')

        plt.show()

        # test_y = scaler_for_y.inverse_transform(test_y)
        #
        # rmse = np.sqrt(mean_squared_error(test_predict, test_y))
        #
        # mae = mean_absolute_error(y_pred=test_predict, y_true=test_y)
        #
        # print('mae:', mae, '   rmse:', rmse)
    return test_predict



def predict_out(time_step=10, train_end=18290):



    scaler_for_x = MinMaxScaler(feature_range=(0, 1))  # ������minmax����
    scaler_for_y = MinMaxScaler(feature_range=(0, 1))


    scaled_x_data = scaler_for_x.fit_transform(data[:, :-1])

    scaled_y_data = scaler_for_y.fit_transform(data[:, -1,np.newaxis])

    # normalized_test_data = scaled_x_data[train_end:]
    test = scaled_x_data[train_end:train_end+time_step,:4]

    test_predict = []
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])

    pred, _ = lstm(X)

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables())
        module_file = tf.train.latest_checkpoint('sm/')
        saver.restore(sess, module_file)



        prob = sess.run(pred, feed_dict={X: [test]})

        predict = prob.reshape((-1,1))

        test_predict.extend(predict)



        test_predict = scaler_for_y.inverse_transform(test_predict)
    return test_predict







# test_predict = predict_out()
# train_lstm()






#
#
#
#
# plt.figure(figsize=(24,8))
#
# plt.plot(data[18000:, -1])
#
# plt.plot([None for _ in range(18290)] + [x for x in test_predict])
#
# plt.show()


# train_lstm()
