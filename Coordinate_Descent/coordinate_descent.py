import os, sys
import math
import numpy as np
from numpy import linalg as LA
import sklearn
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing
import warnings
import random

ERR_THRESH = 0.0001

class CoordinateDescent(object):

    def __init__(self,filename):

        file = np.loadtxt(filename,dtype='float', delimiter=',')
        ones = np.ones((file.shape[0],1))
        file = np.concatenate((file,ones),axis=1)
        self.num_labels = 3

        train_target = file[0:42,0]
        train_target = np.append(train_target, file[60:112,0])
        train_target = np.append(train_target, file[130:164,0])

        test_target = file[42:60,0]
        test_target = np.append(test_target, file[112:130,0])
        test_target = np.append(test_target, file[164:,0])

        train_target = train_target
        test_target = test_target

        self.train_label = np.zeros((train_target.shape[0], 3))
        self.test_label = np.zeros((test_target.shape[0],3))
        self.train_targets = train_target
        self.test_targets = test_target

        for row in range(train_target.shape[0]):
            self.train_label[row,(train_target[row] - 1)] = 1
        for row in range(test_target.shape[0]):
            self.test_label[row,(test_target[row] - 1)] = 1

        self.train_features = np.vstack((file[0:42,1:]))
        self.train_features = np.vstack((self.train_features, file[60:112,1:]))
        self.train_features = np.vstack((self.train_features, file[130:164,1:]))

        self.test_features = file[42:60,1:]
        self.test_features = np.vstack((self.test_features, file[112:130,1:]))
        self.test_features = np.vstack((self.test_features, file[164:,1:]))

        self.feature_index = np.ones(self.train_features.shape[1])
        # self.weight = np.zeros((3,14))
        self.weight = np.random.randn(3,14)

    def pre_process(self):
        # self.train_features[:,:13] = (self.train_features[:,:13] -
        #         np.mean(self.train_features[:,:13],
        #             0))/np.std(self.train_features[:,:13], 0)
        # self.test_features[:, :13] = (self.test_features[:,:13] -
        #         np.mean(self.test_features[:,:13],
        #             0))/np.std(self.test_features[:,:13], 0)

        # self.train_features.astype(np.float32)
        # self.test_features.astype(np.float32)
        self.train_features[:, :13] = preprocessing.scale(self.train_features[:, :13])
        self.test_features[:, :13]  = preprocessing.scale(self.test_features[:, :13])
        print self.test_features
    
    def logistic_regression(self):
        log_reg = linear_model.LogisticRegression(solver="sag", multi_class= 'multinomial', C=100000)
        log_reg.fit(self.train_features, self.train_targets)
        accuracy = 0    
        y_pred = []
        for index in range(self.train_label.shape[0]):
            prediction = log_reg.predict(self.train_features[index])
            y_pred.append(prediction)
            accuracy += (prediction == self.train_targets[index])
        # print log_reg.predict_proba(self.train_features)
        # print y_pred, self.train_targets
        log_loss_ = self.train_label.shape[0] * sklearn.metrics.log_loss(self.train_targets, log_reg.predict_proba(self.train_features))
        print "SKLearn Log Loss: ", log_loss_

        #print "Logistic Regression Prediction Accuracy : " + str(accuracy) + "
        #Log-Likelihood: ", str(log_loss)

    def compute_gradient(self):
        wx_dot = np.dot(self.weight, np.transpose(self.train_features))
        self.prob_x = np.exp(wx_dot)
        tot_prob = np.sum(self.prob_x, 0)
        self.prob_x /= tot_prob
        self.gradient = np.dot(np.transpose(self.train_label) - self.prob_x, self.train_features)

    def random_descent(self, learning_rate, iterations, method="random"):
        for iter_count in range(iterations):
            self.compute_gradient()
            if method == "random":
                rand_col = np.random.choice(np.arange(14))
                rand_row = np.random.choice(np.arange(3))
            elif method == "greedy":
                rand_col = np.argmax(self.gradient) % 14
                rand_row = np.argmax(self.gradient) / 14
            self.weight[rand_row][rand_col] = self.weight[rand_row][rand_col] +\
                                                    learning_rate * self.gradient[rand_row][rand_col]
            curr_cost = 0
            if iter_count % 20 == 0:
                prev_cost = curr_cost if iter_count > 0 else None
                for idx in range(self.train_features.shape[0]):
                    curr_cost += -1.0 * np.log(self.prob_x[self.train_targets[idx]-1][idx])
                if prev_cost is not None and (abs(prev_cost - curr_cost) < ERR_THRESH):
                    return
                print "Accuracy after " + str(iter_count) + " Iterations: " + str(self.predict()) + "  Loss: " + str(curr_cost)
        return

    def predict(self):
        wx_predict = np.dot(self.weight, np.transpose(self.test_features))
        predict = np.exp(wx_predict)
        predict_label = np.argmax(predict, axis = 0)
        accuracy = np.sum(predict_label == self.test_targets-1)
        return accuracy

def main():
    warnings.filterwarnings('ignore')
    coord_descent = CoordinateDescent('wine.data.txt')
    coord_descent.pre_process()
    coord_descent.logistic_regression()

    coord_descent.compute_gradient()
    coord_descent.random_descent(0.01, 50000, method="random")
    coord_descent.predict()

if __name__ == "__main__":
    main()
