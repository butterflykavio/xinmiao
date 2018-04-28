import utils
data_path="DATA.tfrecords"
image=utils.read_and_decode(data_path,batch_size=16)