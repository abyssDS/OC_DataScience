from datetime import datetime
import sys
import os.path

# basic externals
import pandas as pd
import numpy as np
import logging
import warnings
from argparse import ArgumentParser

import cv2
import csv 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend
backend.set_image_dim_ordering('th')

# Imports the -file in parameter into a dataframe. Needs to be in the same format as original CSV from the project
def import_img(import_file):
    # Check if the file exists, otherwise get a new input
    while not import_file or not os.path.isfile(import_file):
        print("File not found", file=sys.stderr)
        import_file = input(
            "Choose an image file > "
        )

    # Importing our data from the file to a dataframe
    print("Importing", import_file, file=sys.stderr)
    img = cv2.imread(import_file)
    img_size =128
    img = cv2.resize(img, (img_size, img_size))
    print(img.shape)
    img = img.reshape(1,3, 128,128)
    print(img.shape)
    return img


def conv_default(filters):
    return Conv2D(filters, kernel_size=3, activation='relu', padding='same')
def pool_default():
    return (MaxPooling2D(pool_size=(2, 2),  strides=(2, 2)))
    
# Transforming our DataFrame to fit our model
def create_model():
  nb_breeds=120
  img_size =128
  color = 1
  model = Sequential()
  model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(color*2 +1, img_size, img_size), padding='same'))
  model.add(pool_default())

  model.add(conv_default(64))
  model.add(conv_default(64))
  model.add(pool_default())

  model.add(conv_default(64))
  model.add(conv_default(64))
  model.add(pool_default())

  model.add(conv_default(128))
  model.add(conv_default(128))
  model.add(pool_default())
            
  model.add(conv_default(128))
  model.add(conv_default(128))
  model.add(pool_default())

  model.add(Flatten()) 
  model.add(Dense(128, activation='relu'))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(nb_breeds, activation='softmax'))

  model.compile(loss='categorical_crossentropy' ,
                optimizer='adam',
                metrics=['accuracy'])
  return model


# Predicts our customers segment (from data X), with our previously generated model (reimported here using joblib)
def predict_breed(model, img):
    pred_classes=model.predict_classes(img)
    
    labels=pd.read_csv('breeds_labels.csv').iloc[:,1]
    
    return labels[pred_classes].item()



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        dest="import_file",
        help="Img file path to predict.",
        metavar="FILE",
    )

    args = parser.parse_args()

    print(
        "This python app will predict the breed of a dog from its picture.",
        file=sys.stderr,
    )
        
    # Importing img
    img = import_img(args.import_file)

    model = create_model()
    
    model
    print(
        "Loading model...",
        file=sys.stderr,
    )
    
    model.load_weights('last_weights.hdf5')
    predicted_breed = predict_breed(model, img)
    
    print(
        "The predicted dog breed on this picture is :",
        file=sys.stderr,
    )
    print(predicted_breed)