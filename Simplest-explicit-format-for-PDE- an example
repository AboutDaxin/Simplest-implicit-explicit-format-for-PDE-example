4 / 1 / 2022
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d   import Axes3D

#参数
tao=1/100
h=1/5
r=tao/h**2

#初始化
u=np.zeros((int(1/h+1),int(1/tao+1)))
for n in range(int(1/h+1)):
    u[n,0]=np.exp(-n*h)

for n in range(int(1/tao+1)):
    u[0,n]=np.exp(n*tao)

for n in range(int(1/tao+1)):
    u[int(1/h),n]=np.exp(n*tao-1)

#递推
for k in range(int(1/tao)):
    for j in range(1,int(1/h)):
        u[j,k+1]=(1-2*r)*u[j,k]+r*(u[j+1,k]+u[j-1,k])

#精确解
u1=np.zeros((int(1/h+1),int(1/tao+1)))
for k in range(int(1/tao+1)):
    for j in range(int(1/h+1)):
        u1[j,k]=np.exp(k*tao-j*h)

#误差
u2=u-u1

#作图
fig=plt.figure()    #创建一个画布
ax=fig.add_subplot(projection='3d')   #创建3D的坐标轴

X=np.arange(0,1+tao,tao)
Y=np.arange(0,1+h,h)

X, Y=np.meshgrid(X, Y)   #依据X，Y创建网格

Z=u2  #公式

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')  #作图
#rstride=  表示横向网格跨度
#cstride=  表示纵向网格跨度



ax.set_zlim(-3*10**-4,0) #设定一下z轴范围

plt.show()

