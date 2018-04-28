import tensorflow as tf
import tensorlayer as tl
import numpy as np


def res_block(x,count,num_filters=128,height=80,width=80):
  prev_nb_channels = x.outputs.get_shape().as_list()[3]
  if count==0:
    shortcut=tl.layers.Conv2dLayer(x,name='extend'+str(count),strides=(1,1,1,1),shape=(1,1,prev_nb_channels,num_filters),padding='SAME',act=tf.nn.relu)
  else:
    shortcut=x
  name='conv1_'+str(count)
  y=tl.layers.Conv2dLayer(x,name=name,strides=(1,1,1,1),shape=(3,3,prev_nb_channels,num_filters),padding='VALID',act=tf.identity)
  name='norm1_'+str(count)
  y=tl.layers.BatchNormLayer(y,epsilon=1e-05,is_train=True,name=name,act=tf.nn.relu)
  name='conv2_'+str(count)
  y=tl.layers.Conv2dLayer(y,name=name,strides=(1,1,1,1),shape=(3,3,prev_nb_channels,num_filters),padding='VALID',act=tf.identity)
  name='norm2_'+str(count)
  y=tl.layers.BatchNormLayer(y,epsilon=1e-05,is_train=True,name=name)
  shortcut = tf.image.resize_image_with_crop_or_pad(shortcut.outputs, target_height=height-count*4,target_width=width-count*4)
  shortcut = tl.layers.InputLayer(shortcut, name='res_input_' + str(count))
  out = tl.layers.ElementwiseLayer([y, shortcut],combine_fn=tf.add,name='merge'+str(count))
  return out



def transform_net(input,batch_size):
    x=input
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        model=tl.layers.InputLayer(inputs=x,name="input")
        model=tf.pad(model.outputs,[[0,0],[40,40],[40,40],[0,0]],"REFLECT")
        model=tl.layers.InputLayer(model,name='input2')
        model=tl.layers.Conv2dLayer(model,act=tf.identity,shape=(9,9,3,32),strides=(1,1,1,1),padding="SAME",name='conv1')
        model=tl.layers.BatchNormLayer(model,epsilon=1e-05,is_train=True,name="norm_1",act=tf.nn.relu)
        model=tl.layers.Conv2dLayer(model,act=tf.identity,shape=(3,3,32,64),strides=(1,2,2,1),padding="SAME",name='conv2')
        model=tl.layers.BatchNormLayer(model,epsilon=1e-05,is_train=True,name="norm_2",act=tf.nn.relu)
        model=tl.layers.Conv2dLayer(model,act=tf.identity,shape=(3,3,64,128),strides=(1,2,2,1),padding="SAME",name='conv3')
        model=tl.layers.BatchNormLayer(model,epsilon=1e-05,is_train=True,name="norm_3",act=tf.nn.relu)
        for i in range(5):
            model=res_block(model,i)
        pic=fractional_conv(model,[batch_size,128,128,64],128,64)
        model=tl.layers.InputLayer(pic,name="input3")
        model=tl.layers.BatchNormLayer(model,epsilon=1e-05,is_train=True,name="norm_4",act=tf.nn.relu)
        pic=fractional_conv(model,outshape=[batch_size,256,256,32],input_kernel=64,output_kernel=32)
        model=tl.layers.InputLayer(pic,name="input4")
        model=tl.layers.BatchNormLayer(model,epsilon=1e-05,is_train=True,name="norm_5",act=tf.nn.relu)
        model=tl.layers.Conv2dLayer(model,act=tf.nn.tanh,shape=(9,9,32,3),strides=(1,1,1,1),padding="SAME",name='conv4')
        picture=tf.multiply(tf.add(model.outputs,1),128)
    return picture

def fractional_conv(input,outshape,input_kernel,output_kernel):
    pic=input.outputs;
    kernel=tf.Variable(tf.random_normal([3,3,output_kernel,input_kernel],stddev=0.01),dtype=tf.float32)
    return tf.nn.conv2d_transpose(pic,kernel,output_shape=outshape,strides=[1,2,2,1])




