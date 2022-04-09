from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation , Dropout
from tensorflow.keras import layers
from kerastuner import tuners
from kerastuner.engine.hyperparameters import HyperParameters
import time
import pickle
from keras.utils import np_utils
import cv2
import os
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import kerastuner 
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import classification_report,confusion_matrix

LOG_DIR = f"{int(time.time())}"
tensorboard = TensorBoard(log_dir=LOG_DIR)

batch_size = 64
img_height = 160
img_width = 160

train_dir='/path/to/train_data_set'
test_dir='/path/to/test_data_set'
validation_dir='/path/to/validation_data_set'

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
valid=get_data(validation_dir)


x_train=[]
y_train=[]

x_test=[]
y_test=[]

x_val=[]
y_val=[]


for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in test:
    x_test.append(feature)
    y_test.append(label)

for feature, label in valid:
    x_val.append(feature)
    y_val.append(label)



y_test = np.array(y_test)
y_train=np.array(y_train)
y_val=np.array(y_val)

x_test=np.array(x_test)
x_train=np.array(x_train)
x_val=np.array(x_val)


from sklearn.utils import shuffle
x_train,y_train=shuffle(x_train,y_train)
x_test,y_test=shuffle(x_test,y_test)
x_val,y_val=shuffle(x_val,y_val)

def build_model(hp):   
    data_augmentation = tf.keras.Sequential([

    tf.keras.layers.experimental.preprocessing.RandomContrast(0.1,1.0)
    ])

    base_model = tf.keras.applications.MobileNetV2(input_shape=(img_height,img_width) + (3,),
                                               include_top=False,
                                               weights='imagenet')
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(img_height,img_width) + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.Flatten()(x)
    for j in range(hp.Int('n_of_dense_layers',1,5)):
        x = tf.keras.layers.Dense(hp.Int(f'Dense_{j}_Size',min_value=8,max_value=256,step=8),activation='relu')(x)
        x = tf.keras.layers.Dropout(hp.Float('Dropout_rate'+str(j), min_value=0.0,max_value=0.5,step=0.00000001))(x)
    outputs = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(loss="binary_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=hp.Float('Learning_Rate',min_value=1e-5,max_value=1e-2,step=0.00000001),
                                 beta_1=hp.Float('beta1',min_value=0.9,max_value=0.999999,step=0.00000001),
                                 beta_2=hp.Float('beta2',min_value=0.9,max_value=0.999999,step=0.00000001)),
                  metrics=["accuracy"])

    return model

tuner = kerastuner.tuners.bayesian.BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=10,  
    executions_per_trial=1,
    directory=LOG_DIR)

tuner.search_space_summary()


tuner.search(x=x_train,
             y=y_train,
             epochs=30,
             batch_size=64,
             callbacks=[tensorboard],
             verbose=1,
             validation_data=(x_val, y_val))

tuner.results_summary() 


with open(f"tuner_{int(time.time())}.pkl", "wb") as f:
    pickle.dump(tuner, f)
