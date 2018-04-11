#### Importing Modules ####
### Third Party Modules ###
import tensorflow as tf
import numpy

class Layer(object):
    def __init__(self, inputs, outputs, input):
        self.biases = tf.Variable(tf.zeros([outputs]))
        init_limit = numpy.sqrt(6. / (inputs + outputs))
        self.weights = tf.Variable(tf.random_uniform([inputs, outputs],
                                                     -init_limit,
                                                     init_limit))
        self.output = tf.nn.sigmoid(tf.add(tf.matmul(input, self.weights),
                                           self.biases))

class Network(object):
    def __init__(self, architecture):
        self.input = tf.placeholder(tf.float32, shape=[None,architecture[0]])
        self.layers = [Layer(architecture[0],architecture[1],self.input)]
        for i in range(1,len(architecture)-1):
            self.layers.append(Layer(architecture[i],architecture[i+1],
                                     self.layers[i-1].output))
        self.output = self.layers[-1].output
