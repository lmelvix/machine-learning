import os, sys
import numpy as np
import random
import matplotlib.pyplot as plt


class ReadFile(object):

    def __init__(self, filename):
        self.file = open(filename, 'r+')
        input_data = self.file.read().split("\n")
        self.train_input = {}
        self.phi_input = {}


        for line in input_data:
            data = line.split()
            if len(data) >= 3:
                self.train_input[int(data[0]),int(data[1]),1] = int(data[2])
                self.phi_input[int(data[0]),int(data[1]),pow(int(data[0]),2),
                               pow(int(data[1]),2),(int(data[0])*int(data[1])),1] = \
                    int(data[2])

class VotingPerceptron(object):

    def __init__(self):
        self.level = 1
        self.confidence = {}
        self.confidence[self.level] = 0
        self.weight = {}
        self.phi_weight = [0,0,0,0,0,0]

    def computeConfidence(self, iteration, input_dict):
        self.weight[self.level] = [0,0,0]
        train_x = list(input_dict.keys())
        for iter in range(iteration):
            random.shuffle(train_x)
            for inp_x in train_x:
                classify = sum(i[0] * i[1] for i in zip(self.weight[self.level],inp_x))
                if classify*input_dict[inp_x] > 0:
                    self.confidence[self.level] += 1
                else:
                    self.level += 1
                    self.confidence[self.level] = 1
                    self.weight[self.level]  = [w+yx for w,yx in zip(self.weight[self.level-1],
                                               [x * input_dict[inp_x] for x in inp_x])]
        print "Computed confidence upto " + str(self.level)

    def plotResult(self, train_input, type):
        x1 = [i[0] for i in train_input.keys()]
        x2 = [i[1] for i in train_input.keys()]
        cluster = []
        color = []
        for keys in train_input.keys():
            if type == 'result':
                result  = self.classify(keys)
                if result > 0:
                    cluster.append(u'+')
                    color.append('b')
                else:
                    cluster.append(u'o')
                    color.append('y')
            else:
                for keys in train_input.keys():
                    if train_input[keys] > 0:
                        cluster.append(u'+')
                        color.append('b')
                else:
                    cluster.append(u'o')
                    color.append('y')

        for x,y,m,c in zip(x1,x2,cluster,color):
                plt.scatter(x, y,marker=m,color=c)
        plt.show()

    def plotConfidence(self):
        y = []
        x = {}
        for i in range(12):
            for j in range(12):
                y.append(self.classify([i,j,1]))
                x[i,j,1] = y
        self.plotResult(x,'result')

    def classify(self, x):
        result = 0
        for level in range(len(self.confidence)):
            temp_wx = sum(i * j for i,j in zip(self.weight[level+1],x))
            temp_c = self.confidence[level+1]
            if temp_wx < 0:
                temp_c = temp_c * -1
            result = result + temp_c
        if result > 0:
            return 1
        else:
            return -1

class KernelPerceptron(VotingPerceptron):

    def computeConfidence(self, iteration, input_dict):
        train_x = list(input_dict.keys())
        for inp_x in train_x:
            classify = sum(i[0] * i[1] for i in zip(self.phi_weight, inp_x))
            print classify
            if classify*input_dict[inp_x] <= 0:
                self.phi_weight = [w+yx for w,yx in zip(self.phi_weight,
                                            [x * input_dict[inp_x] for x in inp_x])]

    def plotResult(self, train_input, type):
        print self.phi_weight
        x1 = [i[0] for i in train_input.keys()]
        x2 = [i[1] for i in train_input.keys()]
        cluster = []
        color = []
        for keys in train_input.keys():
            if type == 'result':
                result  = self.classify(keys)
                if result > 0:
                    cluster.append(u'+')
                    color.append('b')
                else:
                    cluster.append(u'o')
                    color.append('y')
            else:
                for keys in train_input.keys():
                    if train_input[keys] > 0:
                        cluster.append(u'+')
                        color.append('b')
                else:
                    cluster.append(u'o')
                    color.append('y')

        for x,y,m,c in zip(x1,x2,cluster,color):
                plt.scatter(x, y,marker=m,color=c)
        plt.show()

    def plotConfidence(self):
        y = []
        x = {}
        for i in range(12):
            for j in range(12):
                y.append(self.classify([i,j,pow(i,2),pow(j,2),i*j,1]))
                x[i,j,i*i, j*j,i*j,1] = y
        self.plotResult(x,'result')

    def classify(self, x):
        result = sum(i[0] * i[1] for i in zip(self.phi_weight,x))
        if result > 0:
            return 1
        else:
            return -1


def main():
    iterations = 1000
    data_file = ReadFile('data2.txt')
    # vperceptron = VotingPerceptron()
    # vperceptron.computeConfidence(iterations, data_file.train_input)
    # vperceptron.plotConfidence()
    # vperceptron.plotResult(data_file.train_input, 'result')

    kperceptron = KernelPerceptron()
    kperceptron.computeConfidence(iterations, data_file.phi_input)
    kperceptron.plotConfidence()
    # kperceptron.plotResult(data_file.phi_input, 'result')

if __name__ == "__main__":
    main()
