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
cutoff=10 #10
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
df.plot()

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

# 實測
df = pd.read_csv(r".\data\right2.csv")
df = df.iloc[40:-5,:]
data = combine_calibrate(df)
data_center = center(data)
#median_data = pd.DataFrame(median_filter(data_center, 5))
data_filter = pd.DataFrame(freq_filter(data_center, numtaps, cutoff/fs)[:])
#data_center = center(data_filter)
res = pd.DataFrame(combine_threshold(data_filter))
res.columns = ['x','y','z']

plot_freq = 0.05
# 校正

plot_a(data, plot_freq)

# 中值濾波
#plot_a(median_data, 0.1)

# 中心化
plot_a(data_center, plot_freq)
plt.ylim(-0.7,0.7)

# 中值+低通濾波
plot_a(data_filter, plot_freq)
plt.ylim(-0.7,0.7)

# 門檻值
#plot_a(res, plot_freq)
#plt.ylim(-0.7,0.7)

# 積分運算
freq = 0.018 # 積分間格
# 9800 mm/s^2

res["z"] = -res["z"]


# 一次積分速度
v_x = ((9800*res["x"].cumsum() * freq))
plot_v(v_x, plot_freq, 'Vx(mm/s)')
v_y = ((9800*res["y"].cumsum() * freq))
plot_v(v_y, plot_freq, 'Vy(mm/s)')
v_z = ((9800*res["z"].cumsum() * freq))
plot_v(v_z, plot_freq, 'Vz(mm/s)')

# 二次積分位移
dis_x = ((9800*res["x"].cumsum() * freq).cumsum() * freq)
plot_d(dis_x, plot_freq, 'Dx(mm)')
dis_y = ((9800*res["y"].cumsum() * freq).cumsum() * freq)
plot_d(dis_y, plot_freq, 'Dy(mm)')
dis_z = ((9800*res["z"].cumsum() * freq).cumsum() * freq)
plot_d(dis_z, plot_freq, 'Dz(mm)')

# 3D位移
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(dis_x,dis_y,dis_z)
ax.set_xlabel("x(mm)")
ax.set_ylabel("y(mm)")
ax.set_zlabel("z(mm)")
plt.show()