#coding=utf-8

from time import sleep
from threading import Thread
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd
import datetime
from statsmodels.tsa.arima_model import ARIMA
from PIL import Image
import LSTM

#发布段编号
# BMCODE = 61614488002
BMCODE = 61111914003
#显示中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 设置figure_size尺寸
plt.rcParams['figure.figsize'] = (24.0, 12.0)
# 设置背景颜色
# plt.rcParams['figure.facecolor'] = 'b'
# 预测滞后时间
lag_time = 20
#真实值线长
len_figure = 600
#空白区域长度
len_plane = 100
#预测滞后时间
pre_num = 10
#ARIMA参数
p=2;d=1;q=2
#初试预测长度
ori_len =20000
# ori_len =18290
#预测间隔时间(s)
inter_time = 1

#读取数据
# filename = r'.\结果.csv'
# dt = pd.read_csv(filename,index_col='BMCODE')
# dt= dt.iloc[:,:-1]
# dt1 = dt.loc[BMCODE]


filename = r'.\61111914003-lstm.csv'
dt = pd.read_csv(filename)
dt1= dt.iloc[:,-1]



#设置图片
plt.style.use('ggplot')
plt.subplots_adjust(top=0.8,bottom=0.2,left=0.2,right=0.8)
fig = plt.figure(1)
fig.text(0.51,0.9,'上海市交通预测演示系统',bbox=dict(facecolor='#fffe7a', alpha=0.5),color='b',fontsize=30,horizontalalignment='center')
fig.text(0.05,0.5,'发布段编号：\n{0}'.format(BMCODE),bbox=dict(facecolor='#fffe7a', alpha=0.5,capstyle='round'),color='b',fontsize=20)

img = Image.open("./ccc.png")
fig.figimage(img,xo=0.1,yo=1)

ax= fig.add_subplot(2,1,1)
# ax.set_xlabel('时间',fontsize=15)
ax.set_ylabel('速度',fontsize=15)
ax.set_title('ARIMA预测')
ax.set_xlim([0, len_figure + 1+ len_plane+ lag_time/2])
ax.set_ylim([0, 100])


ax1= fig.add_subplot(2,1,2)
ax1.set_xlabel('时间',fontsize=15)
ax1.set_ylabel('速度',fontsize=15)
ax1.set_title('LSTM预测')
ax1.set_xlim([0, len_figure + 1+ len_plane+ lag_time/2])
ax1.set_ylim([0, 100])



#修改索引为时间戳，为使用ARIMA进行预测做准备
date = pd.to_datetime('2017-06-05')
minu2 = datetime.timedelta(minutes=2)
data_index = [date]
for i in range(0,len(dt1)-1):
    date = date+minu2
    data_index.append(date)

dt1.index = data_index
dt1.name = str(dt1.name)

#建立实际速度线条句柄
obsX = list(range(len_figure))
# obsX = list(dt1[:len_figure].index)
data = list(dt1[ori_len:ori_len+len_figure])
line_actual = ax.plot(obsX,data,label='实际速度值',color='b')[0]

line_actual_1 = ax1.plot(obsX,data,label='实际速度值',color='b')[0]


model = ARIMA(data, (p, d, q)).fit()
result = model.forecast(pre_num)

pre_result = result[0]
dt_pre = list(pre_result)
dt_pre.insert(0,dt1[ori_len+len_figure])

#建立预测速度线条句柄
obsX_pre = list(range(len_figure-1,len_figure+pre_num))
# obsX = list(dt1[:len_figure].index)
data_pre = list(dt_pre)
line_pre = ax.plot(obsX_pre,data_pre,label='预测速度值',color='r')[0]

ax.legend(loc='upper right')




pre_result_1 = LSTM.predict_out(1000,ori_len+len_figure)


dt_pre_1 = list(pre_result_1[:pre_num])
dt_pre_1.insert(0,dt1[ori_len+len_figure])

#建立预测速度线条句柄
obsX_pre_1 = list(range(len_figure-1,len_figure+pre_num))
# obsX = list(dt1[:len_figure].index)
data_pre_1 = list(dt_pre_1)
line_pre_1 = ax1.plot(obsX_pre_1,data_pre_1,label='预测速度值',color='r')[0]

ax1.legend(loc='upper right')

lstm_pre = pre_result_1[pre_num:]

#按钮点击事件类
num = 0
len_pre = len(dt1)




class ButtonHandler:
    def __init__(self):
        self.flag=False


    def TheadStart(self):

        for i in range(len_pre-len_figure):
            global num
            try:
                if self.flag==True:
                    sleep(inter_time)
                    obsX.append(len_figure + num+ i)
                    data.append(dt1[ori_len+len_figure + num+ i])
                    obsX.remove(obsX[0])
                    data.remove(data[0])

                    ax.set_xlim([int(obsX[0]),int(obsX[-1])+len_plane+lag_time/2])
                    line_actual.set_xdata(obsX)
                    line_actual.set_ydata(data)

                    ax1.set_xlim([int(obsX[0]), int(obsX[-1]) + len_plane + lag_time / 2])
                    line_actual_1.set_xdata(obsX)
                    line_actual_1.set_ydata(data)

                    # pre_result = Arima(data, pre_num)

                    # t_dt2 = dt1[:ori_len+len_figure + num + i]
                    # model = ARIMA(t_dt2, (p, d, q)).fit()
                    # pre_result = model.forecast(pre_num)

                    # t_dt2 = dt1[:ori_len + len_figure + num + i]

                    # if i == 222:
                    #     print(t_dt2)
                    #     print(data)
                    #     print(dt1[len_figure + num+ i])
                    #
                    # if i == 223:
                    #     print(t_dt2)
                    #     print(data)
                    #     print(dt1[len_figure + num+ i])

                    model = ARIMA(data, (p, d, q)).fit()
                    pre_result = model.forecast(pre_num)


                    obsX_pre.append(len_figure +pre_num+ num+ i)
                    data_pre.append(pre_result[0][pre_num-1])
                    # data_pre.append(p_dt1[0])
                    # p_dt1.remove(p_dt1[0])
                    if len(obsX_pre)>len_figure+pre_num:
                        obsX_pre.remove(obsX_pre[0])
                        data_pre.remove(data_pre[0])

                    line_pre.set_xdata(obsX_pre)
                    line_pre.set_ydata(data_pre)

                    pre_result_1 = lstm_pre[num + i: num + i+pre_num]

                    # pre_result_1 = dt_pre_1[]

                    # pre_result_1 = LSTM.predict_out(pre_num,ori_len+len_figure + num + i)

                    obsX_pre_1.append(len_figure +pre_num+ num+ i)
                    data_pre_1.append(pre_result_1[pre_num-1])
                    # data_pre.append(p_dt1[0])
                    # p_dt1.remove(p_dt1[0])
                    if len(obsX_pre_1)>len_figure+pre_num:
                        obsX_pre_1.remove(obsX_pre_1[0])
                        data_pre_1.remove(data_pre_1[0])

                    line_pre_1.set_xdata(obsX_pre_1)
                    line_pre_1.set_ydata(data_pre_1)

                    plt.draw()
            except:
                print(data)
                break

        num = int(obsX[0])


    def Start(self,even):
        if self.flag==False:
            self.flag=True
            t = Thread(target=self.TheadStart)
            t.start()


    def Stop(self,even):
        self.flag=False


#设置按钮
callback = ButtonHandler()
button_point = plt.axes([0.7,0.05,0.1,0.075])
button_start = Button(button_point,'开始')
button_start.on_clicked(callback.Start)

button_point = plt.axes([0.81,0.05,0.1,0.075])
button_stop = Button(button_point,'暂停')
button_stop.on_clicked(callback.Stop)


# def th():
#     while True:
#         sleep(1)
#         button_time.label = str(datetime.datetime.now())
#         button_time.ax.figure.c
#         # fig.draw()
#
# t_pre= Thread(target=th)
# t_pre.start()


plt.show()





