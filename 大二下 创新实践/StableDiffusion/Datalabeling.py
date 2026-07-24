import os
import glob
import matplotlib.pyplot as plt
from PIL import Image

# 读取图片并生成标签文件
def read_image(root_path='C:/Users/Lenovo/Desktop/datasets', mode='train'):
    """
    读取指定目录下的图片，并根据文件夹名称生成对应的标签文件。
    :param root_path: 数据集的根目录
    :param mode: 数据模式（'train' 或 'test'），用于选择训练集或测试集
    :return: 图片路径列表和标签路径列表
    """
    # 定义类别与标签的映射关系
    img_class = {'buildings': 0,  # 建筑物类别对应标签 0
                 'forest': 1,     # 森林类别对应标签 1
                 'glacier': 2,    # 冰川类别对应标签 2
                 'mountain': 3,   # 山脉类别对应标签 3
                 'sea': 4,        # 海洋类别对应标签 4
                 'street': 5}     # 街道类别对应标签 5

    # 构造数据集路径，例如 'seg_train' 或 'seg_test'
    img_c = 'seg_' + mode
    # 使用 glob 获取所有图片的路径，支持通配符匹配
    # 例如：C:/Users/Lenovo/Desktop/datasets/seg_train/buildings/0.jpg
    img_data = glob.glob(os.path.join(root_path, img_c, '*', '*.jpg'))

    # 初始化存储图片路径和标签的列表
    img_path = []    # 用于保存所有图片的路径
    label_path = []  # 这里未实际使用，可以扩展为保存标签路径

    # 打开文件 data.txt，用于存储图片路径和对应的标签
    with open("data.txt", "w") as f:
        for img in img_data:
            img_path.append(img)  # 将图片路径添加到列表中
            # 获取图片所属的类别名称（即父文件夹名）
            label = os.path.basename(os.path.dirname(img))
            # 根据类别名称获取对应的标签索引
            label_index = img_class[label]
            # 写入文件，格式为 "图片路径 标签"，每行一条
            f.write(img + ' ' + str(label_index) + '\n')

    # 返回图片路径列表和标签路径列表（此处标签路径未使用）
    return img_path, label_path

# 展示图片
def show_imgs(image, col=3):
    """
    展示图片及其对应的类别标签。
    :param image: 图片路径列表
    :param col: 每行展示的图片数量
    """
    num_sample = len(image)  # 获取图片数量
    i = 0  # 初始化计数器
    while i < num_sample:
        img = Image.open(image[i])  # 打开图片
        # 获取图片所属的类别名称（即父文件夹名）
        label = os.path.basename(os.path.dirname(image[i]))
        # 使用 matplotlib 绘制子图
        plt.subplot(int(num_sample / col + 1), col, i + 1)
        plt.imshow(img)  # 显示图片
        plt.title('label:' + label)  # 设置标题为类别标签
        i += 1
    plt.show()  # 显示所有图片

if __name__ == '__main__':
    # 示例图片路径列表（可根据实际图片路径修改）
    img_path = [
        'C:/Users/Lenovo/Desktop/datasets/seg_train/buildings/0.jpg',
        'C:/Users/Lenovo/Desktop/datasets/seg_train/forest/8.jpg',
        'C:/Users/Lenovo/Desktop/datasets/seg_train/mountain/32.jpg'
    ]

    # 展示训练集中的图片及其标签
    show_imgs(img_path)

    # 读取数据集并生成索引文件 data.txt
    read_image(mode='train')
    # 获取生成的 data.txt 文件的绝对路径并打印
    data_abs_path = os.path.abspath("data.txt")
    print(f"data.txt 的绝对路径：{data_abs_path}")