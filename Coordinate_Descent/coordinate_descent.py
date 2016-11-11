import os, sys
import numpy as np
from sklearn import linear_model
import warnings
import random

class CoordinateDescent(object):

    def __init__(self,filename):
        file = np.loadtxt(filename,dtype='float', delimiter=',')
        self.train_target = file[0:49,0]
        self.train_target = np.append(self.train_target, file[59:119,0])
        self.train_target = np.append(self.train_target, file[130:166,0])

        self.test_target = file[49:59,0]
        self.test_target = np.append(self.test_target, file[119:130,0])
        self.test_target = np.append(self.test_target, file[166:,0])

        self.train_features = np.vstack((file[0:49,1:]))
        self.train_features = np.vstack((self.train_features, file[59:119,1:]))
        self.train_features = np.vstack((self.train_features, file[130:166,1:]))

        self.test_features = file[49:59,1:]
        self.test_features = np.vstack((self.test_features, file[119:130,1:]))
        self.test_features = np.vstack((self.test_features, file[166:,1:]))

        self.feature_index = np.ones(self.train_features.shape[1])

    def logistic_regression(self,regularization):
        error = 0.0
        log_reg = linear_model.LogisticRegression(
                                                C=regularization,
                                                solver='newton-cg',
                                                multi_class= 'multinomial')
        log_reg.fit(self.train_features, self.train_target)

        for data in range(self.test_target.shape[0]):
            prediction = log_reg.predict(self.test_features[data,:])
            answer = self.test_target[data]
            if prediction[0] != answer:
                error += 1.0
        print "Logistic Regression Prediction Error : " + \
              str((error/self.test_target.shape[0])*100) + "%"

    def random_descent(self):

        prob_coord = np.array((1./13)*self.feature_index)
        prob_coord[0] = 1
        prob_coord[1:] = 0


        print np.random.choice(np.arange(13), p=prob_coord)




def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    coord_descent = CoordinateDescent('wine.data.txt')
    coord_descent.logistic_regression(0.005)
    coord_descent.random_descent()
if __name__ == "__main__":
    main()