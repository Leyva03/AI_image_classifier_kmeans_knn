__authors__ = ['1633198','1636283','1636526']
__group__ = 'DJ.10'

import numpy as np
import utils
import random
from math import floor
from numpy.linalg import norm

 
class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
         
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        npX = np.array(X)
        
        if npX.dtype != float:
            npX = npX.astype(float)
            
        if npX.ndim > 2:
            shape = npX.shape
            npX = np.reshape(npX, (shape[0] * shape[1], shape[2]))
            
        self.X = npX
        
 
    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################


    def _init_centroids(self):
        """
        Initialization of centroids
        """
    
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        if self.options['km_init'].lower() == 'first':
            self.centroids = [self.X[0].tolist()]
            centroidsIni = 1
            for i in self.X:
                repe = False
                for j in self.centroids:
                    if np.array_equal(i.tolist(), j):
                        repe = True
                if not repe:
                    self.centroids.append(i.tolist())
                    centroidsIni = centroidsIni + 1
                    if centroidsIni == self.K:
                        break
            self.old_centroids = self.centroids.copy()

        elif self.options['km_init'].lower() == 'random_k':
            indices = random.sample(range(len(self.X)), self.K)
            self.centroids = self.X[indices]
            self.old_centroids = self.centroids.copy()
            
        elif self.options['km_init'].lower() == 'second':
            M, N = np.shape(self.X)
            c = np.zeros((self.K, N))

            composite = np.sum(self.X, axis=1)
            ds = np.concatenate((composite[:, np.newaxis], self.X), axis=1)
            ds.sort(axis=0)

            paso = M // self.K

            for j in range(self.K):
                if j == self.K - 1:
                    c[j:] = (np.sum(ds[j * paso:, 1:], axis=0)) / paso
                else:
                    c[j:] = (np.sum(ds[j * paso:(j + 1) * paso, 1:], axis=0)) / paso

            self.centroids = c
            
            
        else:
            self.centroids = np.random.rand(self.K, self.X.shape[1])
            self.old_centroids = np.random.rand(self.K, self.X.shape[1])        
    
        
    def get_labels(self):
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        """
        labels = []
        for pix in self.X:
            distMin = np.inf
            centroid_proper = -1
            for ind in range(len(self.centroids)):
                dist = norm(np.array(pix) - np.array(self.centroids[ind]))
                if dist < distMin:
                    distMin = dist
                    centroid_proper = ind
            labels.append(centroid_proper)
        self.labels = labels
        """
        """
        distances = np.linalg.norm(self.X[:, np.newaxis, :] - self.centroids, axis=2)
    
        #distances = np.linalg.norm(self.X[:, np.newaxis, :] - np.expand_dims(self.centroids, axis=0), axis=2)
        self.labels = np.argmin(distances, axis=1)
        """
        distances = np.linalg.norm(self.X - np.array(self.centroids)[:, np.newaxis], axis=2)
        self.labels = np.argmin(distances, axis=0)

    
    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        self.get_labels()
        self.old_centroids = self.centroids
        centroidNuevo = [[] for l in range(self.K)]
        
        for i in range(len(self.X)):
            centroidNuevo[self.labels[i]].append(self.X[i])       
            
        for j in range(len(centroidNuevo)):
            centroidNuevo[j]=np.average(np.array(centroidNuevo[j]), 0)
            
        self.centroids = centroidNuevo
        

    
    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        #return np.allclose(self.centroids, self.old_centroids,rtol=self.options['tolerance'], atol=self.options['tolerance'])
        return np.array_equal(self.centroids, self.old_centroids)
    
    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        """
        self._init_centroids()
        ite = 0
        chivato = False
        while (chivato == False and ite < self.options['max_iter']): #comprovem si convergeix
            self.get_labels() #trobem quin es el centroide mes proper
            self.get_centroids() #calculem els nous centres
            if self.converges():
                chivato = True
            ite = ite + 1
        """
        self._init_centroids()
        self.get_labels()
        self.get_centroids()
        ite = 1
        chivato = False
        while not chivato and ite < self.options['max_iter']:
            self.old_centroids = self.centroids.copy()
            self.get_labels()
            self.get_centroids()
            chivato = self.converges()
            ite += 1
    
    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """
    
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        dist = distance(self.X, self.centroids)
        x2 = np.square(np.min(dist, axis=1))
        suma = np.sum(x2)
        mult = np.multiply(np.divide(1,self.X.shape[0]), suma)
        
        return mult

    
    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        
        self.K = 2
        self.fit()
        
        last=self.withinClassDistance()
        for i in range(3, max_K + 1):
            self.K = i
            self.fit()
            act = self.withinClassDistance()
            dec = 100 * act / last
            if 100 - dec < 50:
                self.K = i-1
                break
            last = act
            
            
            

def distance(X, C):
        """
        Calculates the distance between each pixel and each centroid
        Args:
            X (numpy array): PxD 1st set of data points (usually data points)
            C (numpy array): KxD 2nd set of data points (usually cluster centroids points)
    
        Returns:
            dist: PxK numpy array position ij is the distance between the
            i-th point of the first set an the j-th point of the second set
        """
    
        #########################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        """
        dist = []
        for p in X:
            distPixels = []
            for c in C:
                distPixels.append(norm(np.array(p) - np.array(c)))
            dist.append(distPixels)
        return dist
        """
        dist = np.linalg.norm(X[:, np.newaxis, :] - C, axis=2)
        return dist
        
    
def get_colors(centroids):
        """
        for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
        Args:
            centroids (numpy array): KxD 1st set of data points (usually centroid points)
    
        Returns:
            labels: list of K labels corresponding to one of the 11 basic colors
        """

        #########################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #########################################################

        labColor = []
        probColors = utils.get_color_prob(centroids)
        for i in range(len(centroids)):
            Hprob = 0
            indice = -1
            for j in range(len(probColors[i])):
                p = probColors[i][j]
                if p > Hprob:
                    Hprob = p
                    indice = j
                    
            labColor.append(utils.colors[indice])
        return labColor        

    