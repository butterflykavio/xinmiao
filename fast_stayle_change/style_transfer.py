import tensorflow as tf
import tensorlayer as tl
import numpy as np
import utils
from Image_Transform_net import transform_net
from vgg19 import VGG19

style_path="style.jpg"
style_pic=utils.load_image(style_path).reshape([1,256,256,3])
STYLE_LAYERS=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
CONTENT_LAYERS=['conv4_2']
loss_ratio=1e-7
max_epoch=1000
batch_size=16
model_path=".\pre_trained_model.\imagenet-vgg-verydeep-19.mat"
data_path="DATA.tfrecords"

content=tf.placeholder(dtype=tf.float32,shape=[batch_size,256,256,3],name="content")
style=tf.constant(style_pic,shape=[1,256,256,3],dtype=tf.float32,name="style")
net=VGG19(model_path)

def gram_matrix(tensor):
    shape = tensor.get_shape()
    num_channels = int(shape[3])
    matrix = tf.reshape(tensor, shape=[-1, num_channels])
    gram = tf.matmul(tf.transpose(matrix), matrix)
    return gram

pic=transform_net(content,batch_size)

content_layers = net.feed_forward(input_image=content, scope='content')
Ps = {}
for id in CONTENT_LAYERS:
    Ps[id] = content_layers[id]
style_layers = net.feed_forward(input_image=style, scope='style')
As = {}
for id in STYLE_LAYERS:
    As[id] = style_layers[id]
Fs = net.feed_forward(pic, scope='pic')

""" compute loss """
L_content = 0
L_style = 0
for id in Fs:
    if id in CONTENT_LAYERS:
        F = Fs[id]
        P = Ps[id]
        _, h, w, d = F.get_shape()
        N = h.value * w.value
        M = d.value
        L_content += tf.reduce_sum(tf.pow((F - P), 2)) / (N * M)

    elif id in STYLE_LAYERS:
        F = Fs[id]
        A = As[id]
        _, h, w, d = F.get_shape()
        N = h.value * w.value
        M = d.value
        FG = gram_matrix(F)
        AG = gram_matrix(A)
        L_style += 0.2 * tf.reduce_sum(tf.pow((FG - AG), 2))/(N*M)
L_total = L_content +  L_style*loss_ratio

with tf.Session() as sess:
    train_op=tf.train.AdamOptimizer().minimize(L_total)
    tf.global_variables_initializer().run()

    train_image=utils.read_and_decode(data_path,batch_size=batch_size)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    step=0
    while not coord.should_stop():
        image_batch=tf.train.shuffle_batch(train_image,batch_size,capacity=1000,min_after_dequeue=1000+batch_size*10,num_threads=5)
        sess.run(train_op)
        if step%0==0:
            print(str(sess.run(L_total)))