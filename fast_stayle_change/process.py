import tensorflow as tf
import utils
import os
import numpy as np
dir='.\data\\val2014'

# utils.resize("style.jpg")
#
# error_image=[]
#
# image_names=utils.get_image_paths('.\\data\\val2014\\')
# for image_name in image_names:
#     image=utils.load_image(image_name)
#     if len(image.shape)<3:
#         error_image.append(image_name)
# for i in error_image:
#     os.remove("G:\git\\fast-transfer-style" + str(i))
#
# utils.rename()
#utils.img_to_tfrecord(dir)