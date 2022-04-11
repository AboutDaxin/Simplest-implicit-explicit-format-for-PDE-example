# 4 / 1 / 2022
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d   import Axes3D
import os
from time import time

#参数
tao=1/100
h1=np.linspace(1/2**10,1/2**2,10)
error=np.zeros((1,10))
T=np.zeros((1,10))  #存储运行时间

for count in range(10):  #分10个步长点进行
    start=time()
    h=h1[count]
    r = tao / h ** 2
    # 小u初值初始化
    u = np.zeros((int(1 / h + 1), int(1 / tao + 1)))
    for n in range(int(1 / h + 1)):
        u[n, 0] = np.exp(-n * h)

    # 构建A1
    upper_diag = np.ones((1, int(1 / h) - 1)) * (-r/2)
    center_diag = np.ones((1, int(1 / h) - 1)) * (1 + r)
    lower_diag = np.ones((1, int(1 / h) - 1)) * (-r/2)
    A1 = np.eye(int(1 / h) - 1) * center_diag
    # 然后通过切片来修改上下两条对角线
    np.fill_diagonal(A1[: -1, 1:], upper_diag)  # 修改上对角线
    np.fill_diagonal(A1[1:, : -1], lower_diag)  # 修改下对角线

    # 构建A2
    upper_diag = np.ones((1, int(1 / h) - 1)) * (r/2)
    center_diag = np.ones((1, int(1 / h) - 1)) * (1 - r)
    lower_diag = np.ones((1, int(1 / h) - 1)) * (r/2)
    A2 = np.eye(int(1 / h) - 1) * center_diag
    # 然后通过切片来修改上下两条对角线
    np.fill_diagonal(A2[: -1, 1:], upper_diag)  # 修改上对角线
    np.fill_diagonal(A2[1:, : -1], lower_diag)  # 修改下对角线

    # 构建A
    A=np.dot(np.linalg.inv(A1),A2)


    # 构建和初始化大U
    U = np.zeros((int(1 / h) - 1, int(1 / tao) + 1))
    U[:, 0] = u[1:int(1 / h) , 0]

    # 构建边界条件b
    b = np.zeros((int(1 / h) - 1, int(1 / tao) + 1))
    for i in range(int(1 / tao) + 1):
        b[0, i] = r/2 * (np.exp(i * tao)+np.exp((i+1) * tao))
        b[int(1 / h) - 2, i] = r/2 * (np.exp(i * tao - 1)+np.exp((i+1) * tao - 1))
        b[:,i]=np.dot(np.linalg.inv(A1),b[:,i])

    #递推U结果
    for i in range(0,int(1 / tao) ):
        U[:, i+1] = np.dot(A,U[:,i])+b[:,i]

    # 完善边界条件及矩阵u
    for n in range(int(1 / tao + 1)):
        u[0, n] = np.exp(n * tao)

    for n in range(int(1 / tao + 1)):
        u[int(1 / h), n] = np.exp(n * tao - 1)

    u[1:int(1 / h), 1:int(1 / tao) + 1] = U[:, 1:]


    # 精确解
    u1 = np.zeros((int(1 / h + 1), int(1 / tao + 1)))
    for k in range(int(1 / tao + 1)):
        for j in range(int(1 / h + 1)):
            u1[j, k] = np.exp(k * tao - j * h)

    # 误差
    u2 = np.abs(u - u1)

    # 求error
    error1 = np.zeros((1, int(1 / tao + 1)))
    for i in range(int(1 / tao + 1)):
        error1[0, i] = np.max(u2[:, i])

    #存储error
    #注，此处的20代表取的时刻
    error[0,count]=error1[0,20]

   #运行时间
    end=time()
    T[0,count]=end-start


#lnh作图
plt.figure()
plt.title('lnh-ln(error)')
X1=np.log(h1)
Y1=np.log(error.reshape(10,))
plt.plot(X1,Y1,marker='+',linestyle='--',color='blue')
plt.xlabel('ln(h)')
plt.ylabel('ln(error)')

#在根目录输出图文件
plt.tight_layout()
plt.savefig('Crank-Nicolson lnh-ln(error) figure')
os.startfile(os.path.join(os.getcwd(),'Crank-Nicolson lnh-ln(error) figure.png'))

#lnt作图
plt.figure()
plt.title('lnt-ln(error)')
X=np.log(T.reshape(10,))
Y=np.log(error.reshape(10,))
plt.plot(X,Y,marker='+',linestyle='--',color='red')
plt.xlabel('ln(t)')
plt.ylabel('ln(error)')

#在根目录输出图文件
plt.tight_layout()
plt.savefig('Crank-Nicolson lnt-ln(error) figure')
os.startfile(os.path.join(os.getcwd(),'Crank-Nicolson lnt-ln(error) figure.png'))


#作误差3D图
fig=plt.figure()    #创建一个画布
ax=fig.add_subplot(projection='3d')   #创建3D的坐标轴

X=np.arange(0,1+tao,tao)
Y=np.arange(0,1+h,h)

X, Y=np.meshgrid(X, Y)   #依据X，Y创建网格

Z=u2  #公式

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')  #作图
plt.xlabel('t')
plt.ylabel('h')


ax.set_zlim(0,np.max(u2)) #设定一下z轴范围

plt.show()


