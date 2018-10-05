# Shared Input Layer
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten,Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

import os
path = 'C:\\Users\\марк\\YandexDisk\\data'
os.chdir(path)

# Каталог с данными для обучения
train_dir = 'train/'
# Каталог с данными для проверки
val_dir = 'val'
# Каталог с данными для тестирования
test_dir = 'test'
# Размеры изображения
img_width, img_height = 150, 150
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
# Количество эпох
epochs = 45
# Размер мини-1045выборки
batch_size = 16
# Количество изображений для обучения
nb_train_samples = 16558
# Количество изображений для проверки
nb_validation_samples = 2152
# Количество изображений для тестирования
nb_test_samples = 2152




# input layer
visible = Input(shape=(input_shape))
# first feature extractor

conv1 = Conv2D(150, kernel_size=4, activation='relu')(visible)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, kernel_size=4, activation='relu')(visible)
pool2 = MaxPooling2D(pool_size=(4, 4))(conv2)

drop1 = Dropout(0.1)(pool2)

conv3 = Conv2D(32, kernel_size=8, activation='relu')(drop1)
pool3 = MaxPooling2D(pool_size=(5, 5))(conv3)
flat1 = Flatten()(pool3)



conv4 = Conv2D(32, kernel_size=4, activation='relu')(pool1)
pool4 = MaxPooling2D(pool_size=(5, 5))(conv4)
drop2 = Dropout(0.1)(pool4)
flat3 = Flatten()(drop2)

conv5 = Conv2D(32, kernel_size=4, activation='relu')(pool1)
pool5 = MaxPooling2D(pool_size=(3, 3))(conv5)
drop3 = Dropout(0.1)(pool5)



conv6 = Conv2D(16, kernel_size=4, activation='relu')(drop2)
pool6 = MaxPooling2D(pool_size=(3, 3))(conv6)
flat2 = Flatten()(pool6)

merge3 = concatenate([flat1, flat2, flat3])
# interpretation layer
hidden1 = Dense(64, activation='relu')(merge3)
drop4 = Dropout(0.2)(hidden1)
# prediction output
output = Dense(1, activation='sigmoid')(drop4)
model = Model(inputs=visible, outputs=output)
# summarize layers
print(model.summary())
# plot graph

plot_model(model, to_file='shared_input_layer.png')

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



datagen = ImageDataGenerator(rescale=1. / 255)



train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')



model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)


scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)


print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))


# Генерируем описание модели в формате json
model_json = model.to_json()
# Записываем модель в файл
json_file = open("pipe_binaryAPI.json", "w")
json_file.write(model_json)
json_file.close()

model.save_weights("pipe_binaryAPI.h5")
