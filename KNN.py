__authors__ = ['1633198','1636283','1636526']
__group__ = 'DJ.10'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """

        self.train_data = train_data.reshape(len(train_data), train_data[0].size).astype(float)


    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        
        dataReshaped = test_data.reshape(len(test_data), test_data[0].size)
        dataRFloat = dataReshaped.astype(float)
        
        distancies = cdist(dataRFloat, self.train_data)
        indexs = np.argsort(distancies)[:, :k]
        self.neighbors = self.labels[indexs]
    
    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """

        mesVotats = []
        percentatge = []

        for i in self.neighbors:
            vots = {}
    
            for j in i:
                vots[j] = vots.get(j, 0) + 1
    
            maxVots = max(vots.values())
            majoria = []
            for vot, cops in vots.items():
                if cops == maxVots:
                    majoria.append(vot)
    
            mesVotats.append(majoria[0])
            percentatge.append(maxVots / len(i))
    
        return np.array(mesVotats, dtype='<U8')


    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the class 2nd the  % of votes it got
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
