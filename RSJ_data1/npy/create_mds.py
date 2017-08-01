# coding: UTF-8

import numpy as np
import scipy as sp
import scipy.stats as ss
import csv
from time import time
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

class chnge_MDS():
	def __init__(self,data,N):
		self.data   = data
		self.N      = N
		#self.point  = np.random.rand(data.shape[0])*100 + 100.0
		clf = manifold.MDS(n_components=1, n_init=1, max_iter=100)
		self.point  =clf.fit_transform(self.data)[:,0]*1000 + 1000.0
		self.error  = []

	def create_map(self):
		for i in range(self.N):
			self.update(self.data)

	def update(self, data ):
		
		fi    = np.zeros(self.point.shape[0])
		for row , i in zip(data , range(self.point.shape[0])):
			for dist , j in zip(row,range(self.point.shape[0])):
				if dist != 1.0:
					fi[i] += (abs(self.point[i]-self.point[j])-dist)*(self.point[j]-self.point[i])/abs(self.point[j]-self.point[i]+0.001)
				
			self.point[i] = self.point[i] + (fi[i] / self.point.shape[0])
		self.error.append(abs(fi).sum())


		

if __name__ == '__main__':
	distance2 = np.load("distance/distance2.npy")
	distance3 = np.load("distance/distance3.npy")
	distance4 = np.load("distance/distance4.npy")
	distance5 = np.load("distance/distance5.npy")
	point2    = np.load("point/point2.npy")
	point3    = np.load("point/point3.npy")
	point4    = np.load("point/point4.npy")
	point5    = np.load("point/point5.npy")

	
	creata_map2 = chnge_MDS(1.0/distance2,100)
	creata_map2.create_map()
	print(np.argsort(creata_map2.point))
	creata_map2.point = (creata_map2.point-creata_map2.point.min())
	creata_map2.point = creata_map2.point/creata_map2.point.max()

	np.save("esimate_point/estimate_point2.npy",creata_map2.point)

	creata_map3 = chnge_MDS(1.0/distance3,100)
	creata_map3.create_map()
	print(np.argsort(creata_map3.point))
	creata_map3.point = (creata_map3.point-creata_map3.point.min())
	creata_map3.point = creata_map3.point/creata_map3.point.max()

	np.save("esimate_point/estimate_point3.npy",creata_map3.point)

	creata_map4 = chnge_MDS(1.0/distance4,100)
	creata_map4.create_map()
	print(np.argsort(creata_map4.point))
	creata_map4.point = (creata_map4.point-creata_map4.point.min())
	creata_map4.point = creata_map4.point/creata_map4.point.max()

	np.save("esimate_point/estimate_point4.npy",creata_map4.point)

	creata_map5 = chnge_MDS(1.0/distance5,100)
	creata_map5.create_map()
	print(np.argsort(creata_map5.point))
	creata_map5.point = (creata_map5.point-creata_map5.point.min())
	creata_map5.point = creata_map5.point/creata_map5.point.max()

	np.save("esimate_point/estimate_point5.npy",creata_map5.point)



