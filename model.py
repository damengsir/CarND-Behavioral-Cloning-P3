import csv
import numpy as np
from scipy import ndimage
import cv2
import math
from sklearn.utils import shuffle

samples = []
with open('/opt/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

'''加载图像和转向角数据到GPU工作的空间'''
def generator(samples,batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                source_path = batch_sample[0]
                filename = source_path.split('\\')[-1]
                current_path = '/opt/data/IMG/' + filename
                image = ndimage.imread(current_path)
                images.append(image)
                measurement = float(batch_sample[3])#######
                measurements.append(measurement)

            '''数据增强'''
            augmented_images,augmented_measurements = [], []
            #返回元组列表
            for image,measurement in zip(images,measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                #1是水平翻转，0是垂直翻转，-1是水平垂直翻转
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)


            '''转化为数组格式，keras需要使用该形式'''
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield shuffle(X_train,y_train)
 
batch_size = 20
train_generator = generator(train_samples,batch_size = batch_size)
validation_generator = generator(validation_samples,batch_size = batch_size)

'''搭建一个简单的神经网络'''
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(36,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(48,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,steps_per_epoch=math.ceil(len(train_samples)/batch_size),validation_data=validation_generator,validation_steps=math.ceil(len(validation_samples)/batch_size), epochs=4,verbose=1)
model.save('model.h5')
