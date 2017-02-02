import struct
import numpy as np
import  matplotlib.pyplot as plt
from PIL import Image

filename='/Users/xujiang/technologies/caffe/data/mnist/train-images-idx3-ubyte'

binfile=open(filename,'rb')
buf=binfile.read()

index=0
magic,numImages,numRows,numColumns=struct.unpack_from('>IIII',buf,index)
index+=struct.calcsize('>IIII')

for image in range(0,numImages):
    im=struct.unpack_from('>784B',buf,index)
    index+=struct.calcsize('>784B')
    im=np.array(im,dtype='uint8')
    im=im.reshape(28,28)
    im=Image.fromarray(im)
    im.save('train/train_%s.bmp'%image,'bmp')
