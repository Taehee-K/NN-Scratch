import numpy as np              # library for linear algebra
import h5py                     # data
from random import randint      # random initialization

#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')               # load data
x_train = np.float32(MNIST_data['x_train'][:])              # train data as numpy array
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))    # train label
x_test = np.float32(MNIST_data['x_test'][:])                # test data as numpy array
y_test = np.int32(np.array(MNIST_data['y_test'][:,0]))      # test label
MNIST_data.close()

class CONV_LAYER:
    def __init__(self, dim_ifmap, num_inch, dim_kernel, num_outch, padding, batch_size):
        self.dim_ifmap = dim_ifmap
        self.num_inch = num_inch
        self.dim_kernel = dim_kernel
        self.num_outch = num_outch
        self.padding = padding

        self.kernels = np.random.randn(num_inch, num_outch, dim_kernel, dim_kernel) / np.sqrt(num_inch * num_outch * dim_kernel * dim_kernel)

        self.batch_size = batch_size

    def forward(self, ifmap): 
        self.dim_ofmap = (self.dim_ifmap - self.dim_kernel + 2 * self.padding) + 1
        # input feature of map padding
        padded_ifmap = np.pad(ifmap, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        # output feature map --> reset to zero
        ofmap = np.zeros((self.batch_size, self.num_outch, self.dim_ofmap, self.dim_ofmap), dtype=float)
        for n in range(self.batch_size):
            for x in range(self.dim_ofmap): # output dimension
                for y in range(self.dim_ofmap): # output dimension
                    for k in range(self.num_outch): # output channel
                        for c in range(self.num_inch):  # input channel
                            for i in range(self.dim_kernel):    # kernel dimension
                                for j in range(self.dim_kernel):    # kernel dimension                 
                                    # output feature map
                                    ofmap[n, k, x, y] += self.kernels[c, k, i, j] * padded_ifmap[n, c, x + i, y + j]

        return ofmap

    def backprop(self, I, dO):

        # padding I
        padded_I = np.pad(I, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        # output
        dK = np.zeros((self.num_inch, self.num_outch, self.dim_kernel, self.dim_kernel), dtype=float)
        
        # convolution
        for n in range(self.batch_size):    
            for x in range(self.dim_ofmap): # output dimension
                for y in range(self.dim_ofmap): # output dimension
                    for k in range(self.num_outch): # output channel
                        for c in range(self.num_inch):  # input channel
                            for i in range(self.dim_kernel):    # kernel dimension
                                for j in range(self.dim_kernel):  # kernel dimension
                                    dK[c, k, i, j] += padded_I[n, c, x + i, y + j] * dO[n, k, x, y]
        
        # convolution
        dI = np.zeros((self.batch_size, self.num_inch, self.dim_ifmap, self.dim_ifmap), dtype=float)
        rotated_kernels = np.rot90(np.rot90(self.kernels, axes=(2,3)), axes=(2,3))
        padded_dO = np.pad(dO, ((0, 0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        for n in range(self.batch_size):
            for x in range(self.dim_ofmap):
                for y in range(self.dim_ofmap):
                    for k in range(self.num_outch):
                        for c in range(self.num_inch):
                            for i in range(self.dim_kernel):
                                for j in range(self.dim_kernel): 
                                    dI[n, c, x, y] += padded_dO[n, k, x + i, y + j] * rotated_kernels[c, k, i, j]
        

        return dK, dI

class FC_LAYER:
    def __init__(self, num_in, num_out):
        self.kernel = np.random.randn(num_in, num_out) / np.sqrt(num_in * num_out)
        self.bias = np.random.randn(1, num_out) / np.sqrt(num_out)
        
    def forward(self, x):
        z = np.dot(x, self.kernel) + self.bias
        return z
    
    def backprop(self, x, dZ2):
        dW = np.dot(x.T, dZ2)
        dZ1 = np.dot(dZ2, self.kernel.T)
        dB = np.sum(dZ2, axis=0, keepdims=True)
        return dW, dZ1, dB

class RELU_LAYER:
    def forward(self, x):
        return x*(x>0)
    def backprop(self, x):
        return 1.0*(x>0)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


class CROSS_ENTROPY_ERROR:
    
    def forward(self, x, y):
        # add small number in case x==0
        return -1.0 * np.sum(np.multiply(np.log(x + 0.001e-10), y))

    def backprop(self, x, y):
        return (x - y)

# mnist dataset 28*28 input image, grayscale(channel = 1)
minibatch_size = 1
conv1 = CONV_LAYER(dim_ifmap=28, num_inch=1, dim_kernel=3, num_outch=5, padding=1, batch_size=minibatch_size)
relu1 = RELU_LAYER()
# output of conv+relu --> input neurons
conv2 = CONV_LAYER(dim_ifmap=28, num_inch=5, dim_kernel=3, num_outch=5, padding=1, batch_size=minibatch_size)
relu2 = RELU_LAYER()
# output of fc1 --> classes of MNIST
fc1 = FC_LAYER(28*28*5, 10)


cse1 = CROSS_ENTROPY_ERROR()    # cross entropy layer
lr = 0.001                      # learning rate
num_epochs = 3                  # epoch
train_iterations = 1000         # 1000 images for training
test_iterations = 100           # 100 images for testing

for epoch in range(num_epochs):
    total_trained = 0       # track number of images trained
    train_correct = 0       # how many correct while training
    train_cost = 0          # track cost function output

    # select random images to train
    rand_indices = np.random.choice(len(x_train)-minibatch_size, train_iterations, replace=True)
    
    for i in rand_indices:
        total_trained += minibatch_size # count number of images trained

        # images flattened in 784 * 1 vector -> need to reshape to 3D input feature map
        # current mini-batch size: 1
        mini_x_train = x_train[i:i + minibatch_size].reshape(minibatch_size,1,28,28)
        mini_y_train = y_train[i:i + minibatch_size]

        # one-hot vectorize label
        one_hot_y = np.zeros((minibatch_size, 10), dtype=float)
        one_hot_y[np.arange(minibatch_size), mini_y_train] = 1.0

        # forward propagation
        conv_ofmap1 = conv1.forward(mini_x_train)
        relu_out1 = relu1.forward(conv_ofmap1)
        conv_ofmap2 = conv2.forward(relu_out1) 
        relu_out2 = relu2.forward(conv_ofmap2) 
        fc_out = fc1.forward(relu_out2.reshape(minibatch_size,28*28*5)) 
        #fc_out = fc1.forward(relu_out1.reshape(minibatch_size,28*28*5))
        prob = softmax(fc_out)
        train_cost += cse1.forward(prob, one_hot_y)
    
        # backpropagation
        dCSE1 = cse1.backprop(prob, one_hot_y)
        dW_FC1, dZ_FC1, dB_FC1 = fc1.backprop(relu_out2.reshape(minibatch_size,28*28*5), dCSE1) 
        #dW_FC1, dZ_FC1, dB_FC1 = fc1.backprop(relu_out1.reshape(minibatch_size,28*28*5), dCSE1)
        dRELU2 = relu2.backprop(conv_ofmap2) 
        dK_CONV2, dI_CONV2 = conv2.backprop(relu_out1, np.multiply(dRELU2, dZ_FC1.reshape(minibatch_size,5,28,28))) 
        dRELU1 = relu1.backprop(conv_ofmap1)
        dK_CONV1, _ = conv1.backprop(dRELU1, np.multiply(dRELU1, dI_CONV2)) 
        #dK_CONV1, _ = conv1.backprop(mini_x_train, np.multiply(dRELU1, dZ_FC1.reshape(minibatch_size,5,28,28)))
        
        # weight update
        conv1.kernels -= lr * dK_CONV1
        conv2.kernels -= lr * np.sum(dK_CONV2) 
        fc1.kernel -= lr * dW_FC1
        fc1.bias -= lr * dB_FC1

        # calculate accuracy
        train_correct += np.sum(np.equal(np.argmax(prob, axis=1), mini_y_train))
        
        # track training
        if (total_trained % 100 == 0):
            print("Trained: ", total_trained, "/", train_iterations*minibatch_size, "\ttrain accuracy: ", train_correct/100, "\ttrain cost: ", train_cost/100)
            train_correct = 0
            train_cost = 0

    # validate after each epoch with test dataset
    test_correct = 0
    for i in range(test_iterations):
        mini_x_test = x_test[i:i+minibatch_size].reshape(minibatch_size,1,28,28)
        mini_y_test = y_test[i:i+minibatch_size]
        # do not shuffle for test images --> testing should be done in order

        # forward propagation
        conv_ofmap1 = conv1.forward(mini_x_test)    # output feature map of conv layer
        relu_out1 = relu1.forward(conv_ofmap1)      # output feature map of relu layer
        conv_ofmap2 = conv2.forward(relu_out1) 
        relu_out2 = relu2.forward(conv_ofmap2) 
        fc_out = fc1.forward(relu_out2.reshape(1,28*28*5)) 
        #fc_out = fc1.forward(relu_out1.reshape(minibatch_size,28*28*5))
        #print(fc_out)
        prob = softmax(fc_out)  # get probability of output using softmax
        test_correct += np.sum(np.equal(np.argmax(prob, axis=1), mini_y_test))  # count how many coorrect
    print ("epoch #: ", epoch, ", test accuracy: ", test_correct/test_iterations/minibatch_size)
