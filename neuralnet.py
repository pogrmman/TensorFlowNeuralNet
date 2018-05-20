#### Importing Modules ####
### Third Party Modules ###
import tensorflow as tf
import numpy

class Layer(object):
    def __init__(self, inputs, outputs, input):
        self.biases = tf.Variable(tf.zeros([outputs]))
        # Initialization from Glorot, Benigo 2010
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
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def train(self, inputs, labels, rate, times, val_inputs=None, val_labels=None):
        self.loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels,
                                                    logits=self.output)
        val = False
        if val_inputs and val_labels:
            self.val_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=val_labels, logits=self.output)
            val = True
        self.optimizer = tf.train.GradientDescentOptimizer(rate)
        if val:
            for i in range(times):
                self.session.run(self.optimizer.minimize(self.loss),
                                 {self.input: inputs})
                if i % 10 == 0:
                    loss = self.session.run(self.val_loss,{self.input: val_inputs})
                    print("Epoch " + str(i) + " Validation Loss is: " + str(loss))
        else:
            for i in range(times):
                _, loss = self.session.run((self.optimizer.minimize(self.loss),
                                            self.loss), {self.input: inputs})
                print("Epoch " + str(i) + " Loss is: " + str(loss))
                    
    def predict(self, input):
        return self.session.run(self.output, {self.input: [input]})
        
