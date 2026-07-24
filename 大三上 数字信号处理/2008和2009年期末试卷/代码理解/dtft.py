import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. 定义信号 ---
a = 0.8
n = np.arange(20)
x_n = a**n

# --- 2. 绘制图 1: 原始信号 x(n) ---
plt.figure(1, figsize=(8, 4))
plt.stem(n, x_n)
plt.title(r'1. 原始信号 (时域): $x(n) = (0.8)^n u(n)$', fontsize=14)
plt.xlabel('n (采样点)', fontsize=12)
plt.ylabel('幅度 $x(n)$', fontsize=12)
plt.grid(True)
plt.savefig("plot1_signal.png")
print("已保存 plot1_signal.png")

# --- 3. 绘制图 2: Z 变换 X(z) (3D "地形") ---
fig2 = plt.figure(2, figsize=(10, 8))
ax = fig2.add_subplot(111, projection='3d')

# 创建一个 z = x + jy 的网格
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# 计算 X(z) = 1 / (1 - a * z^-1) = z / (z - a)
# (我们必须处理 z=a 时的无穷大 "极点")
X_z = Z / (Z - a)

# 计算幅度，并 "裁剪" 无穷大的值，以便绘图
Mag_X_z = np.abs(X_z)
# 裁剪在 10，否则极点处的无穷大会让其他地方都看不见
Mag_X_z_clipped = np.clip(Mag_X_z, 0, 10) 

# 绘制 3D 表面
ax.plot_surface(X, Y, Mag_X_z_clipped, cmap='viridis', edgecolor='none')

# 在 "地板" 上绘制单位圆 (r=1)
theta = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(theta), np.sin(theta), zs=0, zdir='z', color='r', linewidth=3, label='单位圆 (r=1)')
ax.legend()

ax.set_title(r'2. Z 变换 (Z 域): 幅度 $|X(z)|$', fontsize=14)
ax.set_xlabel('Re(z) [x 轴]', fontsize=12)
ax.set_ylabel('Im(z) [y 轴]', fontsize=12)
ax.set_zlabel(r'$|X(z)|$ [海拔/幅度]', fontsize=12)
ax.set_zlim(0, 10) # 设置 "海拔" 限制
plt.savefig("plot2_z_transform.png")
print("已保存 plot2_z_transform.png")

# --- 4. 绘制图 3: DTFT X(e^jω) (2D "波浪") ---
# DTFT 就是 Z 变换在单位圆 (r=1) 上的 "切片"
# z = e^(jω)
w = np.linspace(-np.pi, np.pi, 500)
z_dtft = np.exp(1j * w)

# X(e^jω) = 1 / (1 - a * e^(-jω))
X_dtft = 1 / (1 - a * np.exp(-1j * w))
Mag_X_dtft = np.abs(X_dtft)

plt.figure(3, figsize=(10, 5))
plt.plot(w, Mag_X_dtft, linewidth=3)
plt.title(r'3. 离散时间傅里叶变换 (DTFT): $|X(e^{j\omega})|$', fontsize=14)
plt.xlabel(r'$\omega$ (数字频率 / 角度)', fontsize=12)
plt.ylabel(r'幅度 $|X(e^{j\omega})|$', fontsize=12)
plt.grid(True)
plt.xlim(-np.pi, np.pi)
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
           [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
plt.savefig("plot3_dtft.png")
print("已保存 plot3_dtft.png")