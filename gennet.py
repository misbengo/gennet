from keras.models import Sequential
from keras.layers import Dense, InputLayer
import keras
import math
import numpy
def sortWithIndex(array,number):
    indexes = []
    newArray = array
    newArray.reverse()
    for i in newArray:
        indexes.append(i)
        if len(indexes) == number:
            break
    return indexes
    

class GenNet:
    def __init__(self):
        self.networks = []
    def initialise(self,population, inp, hiddenL, hiddenN, out, outputAc, hiddenAc):
        for i in range(population):
            inputs = keras.layers.Input(shape=(inp,))
            table = []
            for i in range(hiddenL):
                layer = Dense(hiddenN, activation = hiddenAc, name = ('hidden_layer_'+str(len(table))))(inputs)
                if len(table) >= 1:
                    layer = Dense(hiddenN, activation = hiddenAc, name = ('hidden_layer_'+str(len(table))))(table[len(table)-1])
                table.append(layer)
            outputs = Dense(out, activation = outputAc, name = 'output_1')(inputs)
            if len(table) >= 1:
                outputs = Dense(out, activation = outputAc, name = 'output_1')(table[len(table)-1])                                     
            self.networks.append(keras.Model(inputs=inputs,outputs = outputs))
    def predict(self,index,values):
        print(self.networks[index].predict(values))
    def summary(self):
        print(self.networks[0].summary())
    def runGenNet(self, scores):
        if len(scores) < 10:
            return "generation size must be larger than ten in order to rungennet"
        noise = 0.01
        num_breeders = math.ceil((len(self.networks)/10) * 4)
        num_killing = math.floor((len(self.networks)/10) * 6)
        breeders = []
        children = []
        king = self.networks[0]
        for i in range(num_breeders):
            for v in range(num_breeders):
                if i != v:
                    duplicate = True
                    try:
                        numpy.add((self.networks[i].get_weights() , self.networks[v].get_weights()))
                    except ValueError:
                        duplicate = False
                    if not duplicate:
                        child = numpy.add(self.networks[i].get_weights() , self.networks[v].get_weights()) / 2
                        child = child.tolist()
                        children.append(child)
        indexes = sortWithIndex(scores, num_killing)
        for i in indexes:
            self.networks[scores.index(i)].set_weights(children[i])
        

def createGenNet(population, inp, hiddenL, hiddenN, out, outputAc, hiddenAc):
    Network = GenNet()
    Network.initialise(population, inp, hiddenL, hiddenN, out, outputAc, hiddenAc)
    return Network

def sigmoid(x):
    return 1/ (1 + numpy.exp(-x))
