from flask import Flask, jsonify, render_template
import numpy as np
import pandas as pd 
import os
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt 
from sklearn.datasets import load_files
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

app = Flask(__name__)

train_dir = r'C:\Users\likit\OneDrive\Desktop\RVCE\archive\NEU Metal Surface Defects Data\train'
val_dir = r'C:\Users\likit\OneDrive\Desktop\RVCE\archive\NEU Metal Surface Defects Data\valid'
test_dir = r'C:\Users\likit\OneDrive\Desktop\RVCE\archive\NEU Metal Surface Defects Data\test'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train')
def train_models():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(200, 200),
            batch_size=10,
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            val_dir,
            target_size=(200, 200),
            batch_size=10,
            class_mode='categorical')

    resnet_model = build_resnet()
    resnet_history = compile_and_train(resnet_model, train_generator, validation_generator)
    resnet_accuracy = resnet_model.evaluate(validation_generator)

    vggnet_model = build_vggnet()
    vggnet_history = compile_and_train(vggnet_model, train_generator, validation_generator)
    vggnet_accuracy = vggnet_model.evaluate(validation_generator)

    cnn_model = build_cnn()
    cnn_history = compile_and_train_cnn(cnn_model, train_generator, validation_generator)
    cnn_accuracy = cnn_model.evaluate(validation_generator)

    return jsonify({
        'ResNet Accuracy': resnet_accuracy[1],
        'VGGNet Accuracy': vggnet_accuracy[1],
        'CNN Accuracy': cnn_accuracy[1]
    })
def get_data_generators():
    train_dir = r'C:\Users\likit\OneDrive\Desktop\RVCE\archive\NEU Metal Surface Defects Data\train'
    val_dir = r'C:\Users\likit\OneDrive\Desktop\RVCE\archive\NEU Metal Surface Defects Data\valid'
    
    # Define image data generators for training and validation data
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Flow training images in batches of 10 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(200, 200),
        batch_size=10,
        class_mode='categorical')

    # Flow validation images in batches of 10 using validation_datagen generator
    validation_generator = validation_datagen.flow_from_directory(
        val_dir,
        target_size=(200, 200),
        batch_size=10,
        class_mode='categorical')

    return train_generator, validation_generator
@app.route('/history')
def show_training_history():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(200, 200),
            batch_size=10,
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            val_dir,
            target_size=(200, 200),
            batch_size=10,
            class_mode='categorical')

    resnet_model = build_resnet()
    resnet_history = compile_and_train(resnet_model, train_generator, validation_generator)

    vggnet_model = build_vggnet()
    vggnet_history = compile_and_train(vggnet_model, train_generator, validation_generator)

    cnn_model = build_cnn()
    cnn_history = compile_and_train_cnn(cnn_model, train_generator, validation_generator)

    resnet_history_dict = resnet_history.history
    vggnet_history_dict = vggnet_history.history
    cnn_history_dict = cnn_history.history

    return render_template('history.html', 
                            resnet_history=resnet_history_dict,
                            vggnet_history=vggnet_history_dict,
                            cnn_history=cnn_history_dict)
# Model Callback for early stopping
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.98 ):
            print("\nReached 98% accuracy so cancelling training!")
            self.model.stop_training = True
def build_resnet():
    # Define the ResNet model
    model = tf.keras.applications.ResNet50(
        include_top=False,
        input_shape=(200, 200, 3),
        pooling='avg',
        weights='imagenet'
    )

    x = tf.keras.layers.Dense(6, activation='softmax')(model.output)
    resnet_model = tf.keras.Model(model.input, x)
    return resnet_model

    pass

def build_vggnet():
    # Define the VGGNet model
    model = tf.keras.applications.VGG16(
        include_top=False,
        input_shape=(200, 200, 3),
        pooling='avg',
        weights='imagenet'
    )

    x = tf.keras.layers.Dense(6, activation='softmax')(model.output)
    vggnet_model = tf.keras.Model(model.input, x)
    return vggnet_model
    pass

def build_cnn():
    # Define the CNN model
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    return model
    pass
@app.route('/train/cnn')
def train_cnn():
    cnn_model = build_cnn()  # Build the CNN model
    train_generator, validation_generator = get_data_generators()  # Get data generators
    cnn_history = compile_and_train_cnn(cnn_model, train_generator, validation_generator)  # Compile and train the CNN model
    
    return jsonify({'message': 'CNN training started'})
@app.route('/train/vggnet')
def train_vggnet():
    vggnet_model = build_vggnet()
    train_generator, validation_generator = get_data_generators()
    vggnet_history = compile_and_train(vggnet_model, train_generator, validation_generator)
    return jsonify({'message': 'VGGNet training completed'})

@app.route('/train/resnet')
def train_resnet():
    resnet_model = build_resnet()
    train_generator, validation_generator = get_data_generators()
    resnet_history = compile_and_train(resnet_model, train_generator, validation_generator)
    return jsonify({'message': 'ResNet training completed'})

def compile_and_train(model, train_generator, validation_generator):
    # Compile and train the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    print('Compiled!')

    callbacks = myCallback()
    history = model.fit(train_generator,
                        batch_size=32,
                        epochs=20,
                        validation_data=validation_generator,
                        callbacks=[callbacks],
                        verbose=1, shuffle=True)
    return history
    pass

def compile_and_train_cnn(model, train_generator, validation_generator):
    # Compile and train CNN model
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    print('Compiled!')

    callbacks = myCallback()
    history = model.fit(train_generator,
                        batch_size=32,
                        epochs=20,
                        validation_data=validation_generator,
                        callbacks=[callbacks],
                        verbose=1, shuffle=True)
    return history
    pass

if __name__ == '__main__':
    app.run(debug=True)
