import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.utils import np_utils
import cv2
import os
import numpy as np

batch_size = 32
img_height = 160
img_width = 160

train_dir='/path/to/train_data_set'
test_dir='/path/to/test_data_set'

labels = ['Hasta', 'Saglikli']
img_size = 160

def get_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1]
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) 
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)



train=get_data(train_dir)
test=get_data(test_dir)



x_train=[]
y_train=[]

x_test=[]
y_test=[]



for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in test:
    x_test.append(feature)
    y_test.append(label)




y_test = np.array(y_test)
y_train=np.array(y_train)


x_test=np.array(x_test)
x_train=np.array(x_train)


from sklearn.utils import shuffle
x_train,y_train=shuffle(x_train,y_train)
x_test,y_test=shuffle(x_test,y_test)

data_augmentation = tf.keras.Sequential([

tf.keras.layers.experimental.preprocessing.RandomContrast(0.1,1.0)
])

transfer_model = tf.keras.applications.MobileNetV2(input_shape=(img_height, img_width) + (3,),
                                               include_top=False,
                                               weights='imagenet')

transfer_model.trainable = False

inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = transfer_model(x, training=False)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256,activation='relu')(x)
outputs = tf.keras.layers.Dense(1,activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5,beta_1=0.9,beta_2=0.9),
              loss='binary_crossentropy',
              metrics=['accuracy'])

epochs = 40
model.summary()
history = model.fit(x_train,y_train,
                    32,
                    epochs=epochs)

from sklearn.metrics import classification_report,confusion_matrix

predictions = model.predict(x_test)
predictions = tf.where(predictions < 0.5, 0, 1)
print(classification_report(y_test, predictions, target_names = ['hasta (Class 0)','saglikli (Class 1)']))
print('Confusion Matrix')
print(confusion_matrix(y_test,predictions))
