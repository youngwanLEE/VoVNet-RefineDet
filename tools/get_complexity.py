import numpy as np
import matplotlib.pyplot as plt
import time
from caffe.model_libs import *
# display plots in this notebook


plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
#caffe_root = '/home/ywlee/caffe_LYW_old/'  # this file is expected to be in {caffe_root}/examples
import os
#os.chdir(caffe_root)
import sys
import tempfile
#sys.path.insert(0, 'python')
sys.path.append("./python")
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2


model_def =  sys.argv[1]




def _create_file_from_netspec(netspec):
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(str(netspec.to_proto()))
    return f.name


'''
    This is a utility function which computes the 
    complexity of a given network.
    This is slightly modified version from the one written by @fsachin, 
    for his network profile ipython notebook
'''
def get_complexity(net=None, prototxt_file=None):
    # One of netspec, or prototxt_path params should not be None
    #assert (netspec is not None) or (prototxt_file is not None)

#     if netspec is not None:
#         prototxt_file = _create_file_from_netspec(netspec)
#     net = caffe.Net(prototxt_file, caffe.TEST)

    total_params = 0
    total_flops = 0

    net_params = caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt_file).read(), net_params)

    for layer in net_params.layer:
        if layer.name in net.params:

            params = net.params[layer.name][0].data.size
            # If convolution layer, multiply flops with receptive field
            # i.e. #params * datawidth * dataheight
            if layer.type == 'Convolution':  # 'conv' in layer:
                data_width = net.blobs[layer.name].data.shape[2]
                data_height = net.blobs[layer.name].data.shape[3]
                flops = net.params[layer.name][0].data.size * data_width * data_height
                # print >> sys.stderr, layer.name, params, flops
            else:
                flops = net.params[layer.name][0].data.size

            total_params += params
            total_flops += flops



#     if netspec is not None:
#         os.remove(prototxt_file)

    return total_params, total_flops







net = caffe.Net(model_def,      # defines the structure of the model
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

total_params, total_flops = get_complexity(net,model_def)

print "total_param : %d, total_flops : %d" %(total_params, total_flops)
