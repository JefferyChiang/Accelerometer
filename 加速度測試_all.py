from cProfile import label
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate
import os
import sys
import numpy as np 
import math
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. 先校正加速規
# 2. 再利用靜止加速度找出靜止加速度之平均與最大最小門檻值
# 3. 濾波 -> 資料中心化 -> 門檻值 -> 積分

minx = -0.055229175
maxx = 0.474022503
miny = -0.066014847
maxy = 0.475204153
minz = -0.047074322
maxz = 0.488912852
mapmin = -100
mapmax = 100

fs=50 #512
cutoff=40 #10
numtaps=155 #155

def calibrate(data, mind, maxd, minm, maxm):
    dc = ((data - mind) * (maxm-minm) / (maxd-mind) + mapmin)/mapmax
    return dc
def combine_calibrate(df):
    xc = calibrate(df['Voltage'], minx, maxx, mapmin, mapmax)
    yc = calibrate(df['Voltage_0'], miny, maxy, mapmin, mapmax)
    zc = calibrate(df['Voltage_1'], minz, maxz, mapmin, mapmax)
    dc = pd.DataFrame([xc,yc,zc]).T
    dc.columns = ['x', 'y', 'z']
    return dc

def median_filter(data, f_size):
	lgth, num_signal=data.shape
	f_data=np.zeros([lgth, num_signal])
	for i in range(num_signal):
		f_data[:,i]=signal.medfilt(data.iloc[:,i], f_size)
	return f_data
def freq_filter(data, f_size, cutoff):
	lgth, num_signal=data.shape
	f_data=np.zeros([lgth, num_signal])
	lpf=signal.firwin(f_size, cutoff, window='hamming')
	for i in range(num_signal):
		f_data[:,i]=signal.convolve(data.iloc[:,i], lpf, mode='same', method='fft')
	return f_data
def plot_raw(data,fs):
    num_rows, num_cols = data.shape
    fig, ax=plt.subplots()
    labels=['x','y','z']
    color_map=['r', 'g', 'b']
    index=np.arange(num_rows)/fs
    for i in range(3):
        ax.plot(index, data.iloc[:,i], color_map[i], label=labels[i])
    ax.set_xlim([0,num_rows/fs])
    ax.set_xlabel('Time [sec]')
    #ax.set_title('Time domain: '+title)
    ax.legend()

def center(data_filter):
    x_mean, y_mean, z_mean = data_filter.mean()[0], data_filter.mean()[1], data_filter.mean()[2]
    data_center = pd.DataFrame([data_filter.iloc[:,0]-x_mean, data_filter.iloc[:,1]-y_mean, data_filter.iloc[:,2]-z_mean])
    return data_center.T

def thresholding(data, datamin, datamax):
    #datamin, datamax= data.min(), data.max()
    data[(data >= datamin) & (data <= datamax)] = 0
    return data
def combine_threshold(data_center):
    x = data_center[0].copy()
    y = data_center[1].copy()
    z = data_center[2].copy()
    x = thresholding(x, xmin, xmax)
    y = thresholding(y, ymin, ymax)
    z = thresholding(z, zmin, zmax)
    res = pd.DataFrame([x,y,z])
    return res.T

def plot_a(data, t):
    #plt.figure()
    data['t'] = t
    data.columns = ['x', 'y', 'z', 't']
    data['tt'] = data['t'].cumsum()
    data.plot(x='tt',y=['x', 'y', 'z'])
    plt.xlabel('time(s)')
    plt.ylabel('acceleration(g)')

def plot_v(data, t, yl):
    #fig = plt.figure()
    data = pd.DataFrame(data)
    data['t'] = t
    data.columns = ['v', 't']
    data['tt'] = data['t'].cumsum()
    data.plot(x='tt',y='v',legend=False)
    plt.xlabel('time(s)')
    plt.ylabel(yl)

def plot_d(data, t, yl):
    #fig = plt.figure()
    data = pd.DataFrame(data)
    data['t'] = t
    data.columns = ['d', 't']
    data['tt'] = data['t'].cumsum()
    data.plot(x='tt',y='d',legend=False)
    plt.xlabel('time(s)')
    plt.ylabel(yl)

# 讀取靜止加速度
df = pd.read_csv(r".\data\30.csv")

# 校正
data = combine_calibrate(df)
plot_a(data, 0.1)

# 中心化
#data_filter = pd.DataFrame(comb_data[100:-100])
data_center = center(data)
#plot_raw(data_center, fs)
plot_a(data_center, 0.1)
plt.ylim(-0.7,0.7)

# 中值濾波 + 低通濾波
#median_data=median_filter(data, numtaps)
#lpf_data=freq_filter(data, numtaps, cutoff/fs)
comb_data=pd.DataFrame(freq_filter(pd.DataFrame(data_center), numtaps, cutoff/fs))
#plot_raw(comb_data[100:-100], fs)

# 找門檻值

x = comb_data[0]
y = comb_data[1]
z = comb_data[2]
#xmin, xmax = x.min(), x.max()
#ymin, ymax = y.min(), y.max()
#zmin, zmax = z.min(), z.max()
#xmin, xmax = -0.02, 0.02
#ymin, ymax = -0.02, 0.02
#zmin, zmax = -0.02, 0.02
xmin, xmax = -0, 0
ymin, ymax = -0, 0
zmin, zmax = -0, 0
print(xmin, xmax)
print(ymin, ymax)
print(zmin, zmax)

res = combine_threshold(comb_data)
plot_a(res, 0.1)
plt.ylim(-0.7,0.7)

def combine_all(path, plot_freq):
    df = pd.read_csv(path)
    data = combine_calibrate(df)
    data_center = center(data)
    #median_data = pd.DataFrame(median_filter(data, 5))
    data_filter = pd.DataFrame(freq_filter(data_center, numtaps, cutoff/fs)[:])
    res = pd.DataFrame(combine_threshold(data_filter))
    # 校正
    plot_a(data, plot_freq)
    
    # 中心化
    plot_a(data_center, plot_freq)
    plt.ylim(-0.7,0.7)
    # 中值濾波
    #plot_a(median_data, 0.1)

    # 中值+低通濾波
    plot_a(data_filter, plot_freq)
    plt.ylim(-0.7,0.7)

    # 門檻值
    #plot_a(res, 0.1)
    #plt.ylim(-0.7,0.7)
    return res
# 實測
'''
df = pd.read_csv(r".\data\put.csv")
data = combine_calibrate(df)
median_data = pd.DataFrame(median_filter(data, 5))
data_filter = pd.DataFrame(freq_filter(median_data, numtaps, cutoff/fs)[:])
data_center = center(data_filter)
res = pd.DataFrame(combine_threshold(data_center))
'''

plot_freq = 0.05

res_get = combine_all(r".\data\right2.csv",plot_freq)
res_go = combine_all(r".\data\go.csv",plot_freq)
res_put = combine_all(r".\data\put.csv",plot_freq)
plt.close('all')

res_all = combine_all(r".\data\all.csv",plot_freq)
# 校正
#plot_a(data, 0.1)

# 中值濾波
#plot_a(median_data, 0.1)

# 中值+低通濾波
#plot_a(data_filter, 0.1)

# 中心化
#plot_a(data_center, 0.1)
#plt.ylim(-0.7,0.7)

# 門檻值
#plot_a(res, 0.1)
#plt.ylim(-0.7,0.7)
#res = pd.concat([res_get, res_go, res_put],axis=0)
res_get.columns = ['x','y','z']
res_go.columns = ['x','y','z']
#res_go['x'] = 0
res_go['z'] = 0
res_put.columns = ['x','y','z']
res_put['x'] = -res_put['x']

res_get["z"] = -res_get["z"]
res_go["z"] = -res_go["z"]
res_put["z"] = -res_put["z"]
#print(res)


# 積分運算
freq1 = 0.009 #for example freq=10 if you have 10 records per second
freq2 = 0.02
freq3 = 0.016
# 9800 mm/s^2

def v(res, freq):
    return 9800*res["x"].cumsum() * freq, 9800*res["y"].cumsum() * freq, 9800*res["z"].cumsum() * freq

def d(res, freq):
    return ((9800*res["x"].cumsum() * freq).cumsum() * freq), ((9800*res["y"].cumsum() * freq).cumsum() * freq), ((9800*res["z"].cumsum() * freq).cumsum() * freq)

# 一次積分速度
v_x_get, v_y_get, v_z_get = v(res_get, freq1)
v_x_go, v_y_go, v_z_go = v(res_go, freq2)
v_x_put, v_y_put, v_z_put = v(res_put, freq3)

v_x_all = pd.concat([v_x_get, v_x_go+v_x_get.iloc[-1], v_x_put+v_x_get.iloc[-1]+v_x_go.iloc[-1]], axis=0)
v_y_all = pd.concat([v_y_get, v_y_go+v_y_get.iloc[-1], v_y_put+v_y_get.iloc[-1]+v_y_go.iloc[-1]], axis=0)
v_z_all = pd.concat([v_z_get, v_z_go+v_z_get.iloc[-1], v_z_put+v_z_get.iloc[-1]+v_z_go.iloc[-1]], axis=0)
plot_v(v_x_all, plot_freq, 'Vx(mm/s)')
plot_v(v_y_all, plot_freq, 'Vy(mm/s)')
plot_v(v_z_all, plot_freq, 'Vz(mm/s)')


# 二次積分位移
d_x_get, d_y_get, d_z_get = d(res_get, freq1)
d_x_go, d_y_go, d_z_go = d(res_go, freq2)
d_x_put, d_y_put, d_z_put = d(res_put, freq3)
d_x_all = pd.concat([d_x_get, d_x_go+d_x_get.iloc[-1], d_x_put+d_x_get.iloc[-1]+d_x_go.iloc[-1]], axis=0)
d_y_all = pd.concat([d_y_get, d_y_go+d_y_get.iloc[-1], d_y_put+d_y_get.iloc[-1]+d_y_go.iloc[-1]], axis=0)
d_z_all = pd.concat([d_z_get, d_z_go+d_z_get.iloc[-1], d_z_put+d_z_get.iloc[-1]+d_z_go.iloc[-1]], axis=0)
plot_d(d_x_all, plot_freq, 'Dx(mm)')
plot_d(d_y_all, plot_freq, 'Dy(mm)')
plot_d(d_z_all, plot_freq, 'Dz(mm)')

# 3D位移
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(pd.concat([d_x_get, d_x_go+d_x_get.iloc[-1], d_x_put+d_x_get.iloc[-1]+d_x_go.iloc[-1]], axis=0),
        pd.concat([d_y_get, d_y_go+d_y_get.iloc[-1], d_y_put+d_y_get.iloc[-1]+d_y_go.iloc[-1]], axis=0),
        pd.concat([d_z_get, d_z_go+d_z_get.iloc[-1], d_z_put+d_z_get.iloc[-1]+d_z_go.iloc[-1]], axis=0),)
ax.set_xlabel("x(mm)")
ax.set_ylabel("y(mm)")
ax.set_zlabel("z(mm)")
ax.set_xlim(-100,100)
ax.set_ylim(-50,700)
ax.set_zlim(-100,150)
ax.invert_xaxis()

plt.show()
