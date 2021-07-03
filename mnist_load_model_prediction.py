import  tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import scipy
import imageio
import cv2
from PIL import Image
from scipy import ndimage
import scipy.misc
(x, y), (x_val, y_val) = datasets.mnist.load_data()

#%%
# 提取部分数据集中的图片，保存为图片格式
for i in range(20):
    image_array = x_val[i]
    filename = "./images/" + 'mnist_test_%d.jpg' % i
    img = Image.fromarray(np.uint8(image_array))
    img.save(filename)

#%%
# 加载模型
print('loaded model from file.')
network = tf.keras.models.load_model('model.h5', compile=False)
network.compile(optimizer=optimizers.Adam(lr=0.01),
        loss=tf.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

#%%
# 方法一：取数据集中的图片，进行图片大小的处理后，用predict预测

# 取验证集中的某张图片显示
index = 119
plt.imshow(x_val[index])        # （28， 28)
plt.show()
print(y_val[index])

# 对取的图片进行模型所要的格式进行处理
x_pre = x_val[index]
x_pre = tf.cast(x_pre, dtype=tf.float32) / 255.        # （28， 28)
x_pre = tf.reshape(x_pre, [-1, 28*28])        # （1， 784)

# 放入模型预测并输出预测结果
predict = network.predict(x_pre)
predict = np.argmax(predict, axis=1)[0]
print("结果是：" + str(predict))

#%%
# 方法二：取文件夹中的图片，(若为RGB，则按照灰度转换公式，转换为灰度图，如：28*28*3——>28*28)
# 之后同方法一一样处理尺寸大小再预测

# 取文件夹中的图片显示，并灰度处理（3通道的RBG需要灰度处理）
my_image = "my_image2.jpg" # change this to the name of your image file
fname = "images/" + my_image
image = np.array(imageio.imread(fname))         # （28， 28，3)
i1 = image[:, :, 0]     # R通道
i2 = image[:, :, 1]     # G通道
i3 = image[:, :, 2]     # B通道
image = i1*0.299 + i2*0.587 + i3*0.114          # （28， 28)
plt.imshow(image)
plt.show()

# 对取的图片进行模型所要的格式进行处理
image = tf.reshape(image, [-1, 28*28])          # （1， 784)

# 放入模型预测并输出预测结果
predict = network.predict(image)
predict = np.argmax(predict, axis=1)[0]
print("结果是：" + str(predict))

#%%
# 此段for循环程序是等同方法二，仅仅是批量预测图片，并且图片本身为单通道，所以故意写了这一段
for i in range(3):
    filename = "./images/" + 'mnist_test_%d.jpg' % i
    image = np.array(imageio.imread(filename))
    plt.imshow(image)
    plt.show()
    image = tf.reshape(image, [-1, 28 * 28])
    predict = network.predict(image)
    predict = np.argmax(predict, axis=1)[0]
    print("结果是：" + str(predict))