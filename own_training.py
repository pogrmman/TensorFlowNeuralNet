#### Importing Modules ####
### Third-Party Modules ###
import tensorflow as tf
### Package Modules ###
import neuralnet

class OwnTrainNet(neuralnet.Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = []
        self.biases = []
        for layer in self.layers:
            self.weights.append(layer.weights)
            self.biases.append(layer.biases)
            
    def train(self, train_data, rate, batch_size, epochs):
        self._train_setup(train_data, batch_size)
        self.loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.labels,
                                                    logits=self.output)
        self.optimizer = TrainDriver(self, rate)
        self._train(epochs)

    def _train(self, epochs):
        for i in range(epochs):
            batch = self.session.run(self.batch)
            inputs = batch["inputs"]
            outputs = batch["outputs"]
            self.optimizer.compute_gradients()
            self.session.run((self.optimizer.weight_grads, self.optimizer.bias_grads),
                             {self.input: inputs, self.labels: outputs})
            self.optimizer.apply_gradients()
            _, _, loss = self.session.run((self.optimizer.update_w, self.optimizer.update_b, self.loss),
                                       {self.input: inputs, self.labels: outputs})
            print("Epoch " + str(i) + " Loss is: " + str(loss))
        
class TrainDriver(object):
    def __init__(self, net, rate, l2=0, momentum=0):
        self.net = net
        self.loss_func = self.net.loss
        self.rate = rate

    def compute_gradients(self):
        self.weight_grads = tf.gradients(xs=self.net.weights, ys=self.loss_func)
        self.bias_grads = tf.gradients(xs=self.net.biases, ys=self.loss_func)

    def modify_gradients(self):
        pass
    
    def apply_gradients(self):
        for grad in self.weight_grads:
            grad = tf.scalar_mul(self.rate, grad)
        self.update_w = [weights.assign(weights - grads) for weights, grads in zip(self.net.weights, self.weight_grads)]
        for grad in self.bias_grads:
            grad = tf.scalar_mul(self.rate, grad)
        self.update_b = [biases.assign(biases - grads) for biases, grads in zip(self.net.biases, self.bias_grads)]
        
