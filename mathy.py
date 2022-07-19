import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
import random
import os

# for analysis
Pred = []




dir = os.getcwd()


def load_data():
    print("#LOADING TRAINING DATA")
    paths = [r'\train\0', r'\train\1',
             r'\train\2', r'\train\3',
             r'\train\4', r'\train\5',
             r'\train\6', r'\train\7',
             r'\train\8', r'\train\9']

    A = []
    for i in range(10):
        os.chdir(dir + paths[i])
        for file in os.listdir(dir + paths[i]):
            A.append([])
            image = Image.open(file)
            image_array = np.array(image).flatten()
            A[-1].append(image_array)
            A[-1].append(i)

    random.shuffle(A)
    print("ENDED LOADING TRAINING DATA")
    os.chdir(dir)
    return A
def load_test():
    print("#LOADING TEST DATA")
    paths = [r'\test\0', r'\test\1',
             r'\test\2', r'\test\3',
             r'\test\4', r'\test\5',
             r'\test\6', r'\test\7',
             r'\test\8', r'\test\9']

    A = []

    for i in range(10):
        os.chdir(dir + paths[i])
        for file in os.listdir(dir + paths[i]):
            A.append([])
            image = Image.open(file)
            image_array = np.array(image).flatten()
            A[-1].append(image_array)
            A[-1].append(i)


    random.shuffle(A)
    print("ENDED LOADING TEST DATA")
    os.chdir(dir)
    return A

def sigmoid(x): return 1/(1 + np.exp(-x))

class Network:

    def __init__(self, NOL, NPL):
        #NOL: NUMBER OF LAYERS     NPL: NEURONS PER LAYER (ARRAY)
        self.NOL = NOL
        self.NPL = NPL
        self.Layers = []
        #STORES THE A ARRAYS being the inputs, THE DERIVATIVES OF THE SIGMOIDS(TECHNICALLY A INPUTS AFTER L = 0 in another form), and BACK PROPAGATION DATA, being used as in the "delta rule"
        self.A = []
        self.Biases = []
        self.dSigmoids = []
        self.BPD = []


    def Init_Layers(self):
        layers = []
        #runs through all layers except the last one to generate arrays with their weights
        for i in range(self.NOL - 1):
            layers.append([])
            #runs through all neurons in the layer creating the amount of weights equivalent to the size of the next layer
            for j in range(self.NPL[i]):
                #array of weights with size equivalent to the size of next layer
                neuron_weights = np.random.uniform( high=2, low = -2, size=(self.NPL[i + 1], 1))
                layers[i].append(neuron_weights)

        for i in range(self.NOL - 1):
            self.Biases.append(0)
        self.Layers = layers


    def Init_tools(self):
        #creates empty arrays for the functions cited in the __init__ function, they must be cleared every propagation, for new data to be processed
        dS = []
        bp = []
        a = []
        for i in range(self.NOL - 1):
            a.append(np.zeros((self.NPL[i], 1)))
            dS.append(np.zeros((self.NPL[i + 1], 1)))
            bp.append(np.zeros((self.NPL[i + 1], 1)))
        a.append(np.zeros((self.NPL[-1], 1)))
        self.A = a
        self.dSigmoids = dS
        self.BPD =bp


    def Propagate(self, data):

        for i in range(self.NOL - 1):
            for j in range(self.NPL[i]):
                #iterates through every neruon using it to compound sum to the next layer A
                self.A[i+1] += ((self.A[i][j] * self.Layers[i][j]))

            # calculates the activation function (input to next layer) and stores it's derivative in the dsigmoids array
            for k in range(self.NPL[i + 1]):

                self.A[i+1][k] = sigmoid(self.A[i+1][k] + self.Biases[i])
                self.dSigmoids[i][k] = self.A[i+1][k] * (1 - self.A[i+1][k])


    def Bpropagate(self, cost, learning_rate):

        #delta 1, it's more convenient to store it here instead of including conditionals in the loop
        self.BPD[-1] = cost * self.dSigmoids[-1]

        for i in range(self.NOL - 2, 0, -1):
            for j in range(self.NPL[i]):
                #calculate the backprogation factor for every neuron, the BPD indexes are respective to the factor of each neuron
                self.BPD[i - 1][j] = np.sum(self.BPD[i] * self.Layers[i][j])

            #completes calculation for the next delta by multiplying for the sigmoid
            self.BPD[i - 1] = self.BPD[i - 1] * self.dSigmoids[i - 1]

        #actualizes the weights
        for i in range(self.NOL - 1):
            for j in range(self.NPL[i]):
                self.Layers[i][j] -= learning_rate * self.BPD[i] * self.A[i][j]
            self.Biases[i] -= learning_rate * np.sum(self.BPD[i])/len(self.BPD[i])


    def test(self, Data):
        Tots = 0
        Corrects = 0
        random.shuffle(Data)
        for i in range(int(len(Data)/60)):

            self.Init_tools()
            for j in range(self.NPL[0]):
                self.A[0][j] = Data[i][0][j] / 255

            label = Data[i][1]
            Expect_vector = np.zeros((10, 1))
            Expect_vector[label] = 1

            self.Propagate(Data)
            cost = self.A[-1] - Expect_vector

            best = 0
            score = 0
            for k in range(10):
                if self.A[-1][k][0] > score:
                    score = self.A[-1][k][0]
                    best = k
            if best == label: Corrects += 1
            Tots += 1


        print("PRECISION<#TEST_SET>: " + str(Corrects / Tots)), Pred.append(Corrects / Tots)

    def train(self, learning_rate, Iterations, Batch_Size, Data, Test_Data):
        print("TRAINING IS STARTING...")
        Tots = 0
        Corrects = 0
        for run in range(Iterations):
            for i in range(Batch_Size):

                if i % int(Batch_Size * 0.05) == 0:
                    print("======================================")
                    print("Batch: {} Iteration: {}".format(i, run))
                    self.test(Test_Data)


                self.Init_tools()
                for j in range(self.NPL[0]):
                    self.A[0][j] = Data[i][0][j]/255

                label = Data[i][1]
                Expect_vector = np.zeros((10, 1))
                Expect_vector[label] = 1

                self.Propagate(Data)
                cost = self.A[-1] - Expect_vector


                self.Bpropagate(cost, learning_rate)



# CHOSSE NUMBER OF LAYERS (#) AND AMOUNT OF NEURONS PER LAYER([784, 20, 10])
AI = Network(3, [784, 20, 10])
AI.Init_Layers()

# LOAD DATA IN
Data = load_data()
Test_Data = load_test()

#FIRST PARAMETER IS THE LEARNING RATE
AI.train(0.1, 1, 50000, Data, Test_Data)

print("EXPORTING MODEL TO CSV")
df = pd.DataFrame(AI.Layers)
df['Biases'] = AI.Biases
df.to_csv('NN.csv', index=False)


B = []
for i in range(len(Pred)):
   B.append(i + 1)

plt.plot(B, Pred)
plt.show()
