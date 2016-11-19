import os, sys
import math
import numpy as np
from numpy import linalg as LA
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
import warnings
import random


class CoordinateDescent(object):

    def __init__(self,filename):

        file = np.loadtxt(filename,dtype='float', delimiter=',')
        ones = np.ones((file.shape[0],1))
        file = np.concatenate((file,ones),axis=1)
        self.num_labels = 3

        train_target = file[0:49,0]
        train_target = np.append(train_target, file[59:119,0])
        train_target = np.append(train_target, file[130:166,0])

        test_target = file[49:59,0]
        test_target = np.append(test_target, file[119:130,0])
        test_target = np.append(test_target, file[166:,0])

        self.train_label = np.zeros((train_target.shape[0], 3))
        self.test_label = np.zeros((test_target.shape[0],3))
        self.train_targets = train_target
        self.test_targets = test_target

        for row in range(train_target.shape[0]):
            self.train_label[row,(train_target[row]-1)] = 1
        for row in range(test_target.shape[0]):
            self.test_label[row,(test_target[row]-1)] = 1

        self.train_features = np.vstack((file[0:49,1:]))
        self.train_features = np.vstack((self.train_features, file[59:119,1:]))
        self.train_features = np.vstack((self.train_features, file[130:166,1:]))

        self.test_features = file[49:59,1:]
        self.test_features = np.vstack((self.test_features, file[119:130,1:]))
        self.test_features = np.vstack((self.test_features, file[166:,1:]))

        self.feature_index = np.ones(self.train_features.shape[1])

    def predict_function(self, weight, feature):
        wx_dot = (np.dot(weight,feature))
        prob_x = [math.exp(x) for x in wx_dot]
        tot_prob = np.sum(prob_x)
        prob_x = [x/tot_prob for x in prob_x]
        return np.argmax(prob_x)

    def loss_function(self, weight, feature, target):
        wx_dot = (np.dot(weight,feature))
        # print str(feature) + " " + str(weight) + " " + str(wx_dot)
        # print str(wx_dot)

        prob_x = [math.exp(x) for x in wx_dot]
        tot_prob = np.sum(prob_x)

        prob_x = [x/tot_prob for x in prob_x]
        log_loss = -1*math.log(prob_x[int(target)-1])
        return log_loss

    def logistic_regression(self):
        log_reg = linear_model.LogisticRegression(solver="lbfgs", multi_class= 'multinomial')
        log_reg.fit(self.train_features, self.train_targets)
        overall_loss = 0
        for data in range(self.test_label.shape[0]):
            overall_loss += self.loss_function(
                      log_reg.coef_, self.test_features[data], self.test_targets[data])
        print "Logistic Regression Prediction Loss : " + str(overall_loss)

    def random_descent(self,learning_rate, iterations):
        random_weight = np.zeros((3,14))
        for iter_count in range(iterations):
            rand_col = np.random.choice(np.arange(14))
            rand_row = np.random.choice(np.arange(3))
            overall_loss = 0
            for data in range(self.train_targets.shape[0]):
                loss = self.loss_function(random_weight,
                                          self.train_features[data],
                                          self.train_targets[data])
                random_weight[rand_row][rand_col] = random_weight[rand_row][rand_col] +\
                                                    learning_rate * loss
                overall_loss += loss

            if iter_count % 10 == 0:
                print "Iteration : " + str(iter_count) + " Loss : " + \
                      str(overall_loss)


def main():
    warnings.filterwarnings('ignore')
    coord_descent = CoordinateDescent('wine.data.txt')
    coord_descent.logistic_regression()
    coord_descent.random_descent(0.001,1000)

if __name__ == "__main__":
    main()