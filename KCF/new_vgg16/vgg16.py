# VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
# License: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl

from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.nonlinearities import softmax

import lasagne.layers
import pickle
#as3:/usr/local/lib/python3.5/site-packages# cat sitecustomize.py

import theano
import theano.tensor as T

from theano.tensor.nnet.abstract_conv import bilinear_upsampling


with open('new_vgg16/vgg16.pkl', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    vgg_info = u.load()
#vgg_info = pickle.load(open('new_vgg16/vgg16.pkl'))
def build_model(DEFAULT_PAD = 1):
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=DEFAULT_PAD, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=DEFAULT_PAD, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=DEFAULT_PAD, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=DEFAULT_PAD, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=DEFAULT_PAD, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=DEFAULT_PAD, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=DEFAULT_PAD, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=DEFAULT_PAD, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=DEFAULT_PAD, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=DEFAULT_PAD, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=DEFAULT_PAD, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=DEFAULT_PAD, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=DEFAULT_PAD, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    lasagne.layers.set_all_param_values(net['prob'], vgg_info['param values'])

    return net




def get_layer_output_function(layer_name, DEFAULT_PAD = 1, bilinear_up = None):
    vgg_net = build_model(DEFAULT_PAD = DEFAULT_PAD)

    image_batch = T.tensor4('images') # expected shape: (None, 3, 224, 224)

    mean_values = vgg_info['mean value'].astype(theano.config.floatX)

    image_batch_subtracted = image_batch[:,::-1] - mean_values[None, :, None, None]

    output_tensor = lasagne.layers.get_output(vgg_net[layer_name], inputs = image_batch_subtracted )

    if bilinear_up is not None:
        output_tensor = bilinear_upsampling(output_tensor, bilinear_up, use_1D_kernel=False)

    func = theano.function([image_batch], output_tensor)

    return func

classifier = None

def classify_an_image(image_path):
    import skimage.io
    import skimage.transform
    import numpy

    image = skimage.io.imread(image_path)

    print (image.shape)

    center = (image.shape[0]//2, image.shape[1]//2)
    image = skimage.transform.resize(image, (256, 256, 3))

    if numpy.max(image) <= 1.:
        image = (image*255.).astype(theano.config.floatX)

    image = image.transpose(2,0,1)
    image = image[None,:,:,:]

    if classifier is None:
        classifier = get_layer_output_function('prob')

    probs = classifier(image)[0]

    sorted_class_id = numpy.argsort(-probs)

    print ("Top 5 classes:")
    for i in range(5):
        print ("\t\t{}, prob : {}".format(
                        vgg_info['synset words'][sorted_class_id[i]], 
                        probs[sorted_class_id[i]]
                    ))






