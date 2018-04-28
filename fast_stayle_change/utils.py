# Most code in this file was borrowed from https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/15_Style_Transfer.ipynb

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import os
import tensorflow as tf
"""Helper-functions for image manipulation"""
# This function loads an image and returns it as a numpy array of floating-points.
# The image can be automatically resized so the largest of the height or width equals max_size.
# or resized to the given shape
def load_image_batch(filename_list):
    images=np.zeros([filename_list.shape[0],256,256,3])
    for name in range(filename_list.shape[0]):
        images[name] = PIL.Image.open('.\\data\\val2014\\'+str(filename_list[name])+'.jpg')
    return np.float32(images)

def load_image(filename, shape=None, max_size=None):
    image = PIL.Image.open(filename)

    if max_size is not None:
        factor = float(max_size) / np.max(image.size)
        size = np.array(image.size) * factor
        size = size.astype(int)
        image = image.resize(size, PIL.Image.LANCZOS)

    if shape is not None:
        image = image.resize(shape, PIL.Image.LANCZOS)
    return np.float32(image)

def save_image(image, filename):
    image = np.clip(image, 0.0, 255.0)
    image = image.astype(np.uint8)
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')

def plot_images(content_image, style_image, mixed_image):
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    ax = axes.flat[0]
    ax.imshow(content_image / 255.0, interpolation='sinc')
    ax.set_xlabel("Content")
    ax = axes.flat[1]
    ax.imshow(mixed_image / 255.0, interpolation='sinc')
    ax.set_xlabel("Output")
    ax = axes.flat[2]
    ax.imshow(style_image / 255.0, interpolation='sinc')
    ax.set_xlabel("Style")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def get_image_paths( directory ):
    return [x.path for x in os.scandir( directory ) if x.name.endswith(".jpg") or x.name.endswith(".png") ]

def resize(directory):
    if directory.endswith(".jpg") or directory.endswith(".png"):
        image=load_image(directory,shape=[256,256])
        save_image(image,directory)
    else:
        images=get_image_paths(directory)
        for image_file in images:
            image=load_image(image_file)
            save_image(image,image_file)


def rename():
    fileList = os.listdir('.\data\\val2014')
    name=0
    for fileName in fileList:
        os.rename('.\data\\val2014\\'+fileName,'data\\val2014\\'+str(name)+'.jpg')
        name+=1

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def img_to_tfrecord(data_path):
    writer = tf.python_io.TFRecordWriter('DATA.tfrecords')
    for i in range(40438):
        img_name = str(i)+".jpg"
        img_path = data_path +"\\"+ img_name
        img =PIL.Image.open(img_path)
        #img_raw = img.tostring()
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={'image_raw': _bytes_feature(img_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


def read_and_decode(data_path,batch_size):
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=None,capacity=batch_size)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'image_raw': tf.FixedLenFeature([], tf.string)})
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image=tf.reshape(image,[256,256,3])
    image = tf.cast(image, tf.float32)
    return image