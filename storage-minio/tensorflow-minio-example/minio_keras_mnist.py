from minio import Minio
from minio.error import S3Error
import certifi
import urllib3
from urllib.parse import urlparse
import os
import math
from minio.commonconfig import REPLACE, CopySource
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import os, re, math, json, shutil, pprint
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
print("Tensorflow version " + tf.__version__) #Tensorflow version 1.14.0


AUTO = tf.data.experimental.AUTOTUNE

def read_label(tf_bytestring):
    label = tf.io.decode_raw(tf_bytestring, tf.uint8)
    label = tf.reshape(label, [])
    label = tf.one_hot(label, 10)
    return label
  
def read_image(tf_bytestring):
    image = tf.io.decode_raw(tf_bytestring, tf.uint8)
    image = tf.cast(image, tf.float32)/256.0
    image = tf.reshape(image, [28*28])
    return image
  
def load_dataset(image_file, label_file):
    imagedataset = tf.data.FixedLengthRecordDataset(image_file, 28*28, header_bytes=16)
    imagedataset = imagedataset.map(read_image, num_parallel_calls=16)
    labelsdataset = tf.data.FixedLengthRecordDataset(label_file, 1, header_bytes=8)
    labelsdataset = labelsdataset.map(read_label, num_parallel_calls=16)
    dataset = tf.data.Dataset.zip((imagedataset, labelsdataset))
    return dataset 
  
def get_training_dataset(image_file, label_file, batch_size):
    dataset = load_dataset(image_file, label_file)
    dataset = dataset.cache()  # this small dataset can be entirely cached in RAM, for TPU this is important to get good performance from such a small dataset
    dataset = dataset.shuffle(5000, reshuffle_each_iteration=True)
    dataset = dataset.repeat() # Mandatory for Keras for now
    dataset = dataset.batch(batch_size, drop_remainder=True) # drop_remainder is important on TPU, batch size must be fixed
    dataset = dataset.prefetch(AUTO)  # fetch next batches while training on the current one (-1: autotune prefetch buffer size)
    return dataset
  
def get_validation_dataset(image_file, label_file):
    dataset = load_dataset(image_file, label_file)
    dataset = dataset.cache() # this small dataset can be entirely cached in RAM, for TPU this is important to get good performance from such a small dataset
    dataset = dataset.batch(10000, drop_remainder=True) # 10000 items in eval dataset, all in one batch
    dataset = dataset.repeat() # Mandatory for Keras for now
    return dataset


def main():
    BATCH_SIZE = 128
    EPOCHS = 10

    #minio-zyajing
    minio_address = "10.117.233.4:443"
    minio_access_key = "UuWIFVNXcadUsSdn"
    minio_secret_key = "c6Aj3LnJphHUlb2Q2AostJbZPhWcFEOz"
    minio_bucket = "datasets"

    os.environ["AWS_ACCESS_KEY_ID"] = minio_access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = minio_secret_key
    os.environ["AWS_REGION"] = "us-east-1"
    os.environ["S3_ENDPOINT"] = minio_address
    os.environ["S3_USE_HTTPS"] = '1'   # default access by https. if seting S3_USE_HTTPS=0, using http. 
    os.environ["S3_VERIFY_SSL"] = '0'  # If using https, set S3_VERIFY_SSL=0 to disable ssl verify.
    os.environ["AWS_LOG_LEVEL"] = '3'  # Set the logging level for AWS to disable a lot of logs showing.

    training_images_file = 's3://datasets/train-images-idx3-ubyte'
    training_labels_file = 's3://datasets/train-labels-idx1-ubyte'
    validation_images_file = 's3://datasets/t10k-images-idx3-ubyte'
    validation_labels_file = 's3://datasets/t10k-labels-idx1-ubyte'


    ## instantiate the datasets
    training_dataset = get_training_dataset(training_images_file, training_labels_file, BATCH_SIZE)
    validation_dataset = get_validation_dataset(validation_images_file, validation_labels_file)
    print(training_dataset)
    print(validation_dataset)

    #define Keras model
    model = tf.keras.Sequential(
    [
      tf.keras.layers.Input(shape=(28*28,)),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    # print model layers
    print(model.summary())

    #Train and validate the model
    steps_per_epoch = 60000//BATCH_SIZE  # 60,000 items in this dataset
    print("Steps per epoch: ", steps_per_epoch)

    model.fit(training_dataset, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, validation_data=validation_dataset, validation_steps=1)
    
    # model.save(f"s3://datasets/mnist_model")
    model.save("mnist_model") °

if __name__ == "__main__":
    try:
        main()
    except S3Error as exc:
        print("error occurred.", exc)


