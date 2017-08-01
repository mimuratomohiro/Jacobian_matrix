# coding: UTF-8

import numpy as np
import scipy as sp
import scipy.stats as ss
import matplotlib.pyplot as plt
import csv
from time import time

from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)


def create_point(alpha1,alpha2,number):
	point  = np.sort(np.random.beta(alpha1, alpha2, number))*4
	return point
	

def sensor_result(points,sensor,hand_size):
	distance  = np.ones((sensor.size,sensor.size))
	result    = []
	for point in points:
		double_touch_point = np.zeros(sensor.size)
		for i in range(sensor.size):
			if(point + hand_size > sensor[i] and sensor[i] > point - hand_size ):
				double_touch_point[i] = 1
		result.append(double_touch_point)

		for i in range(sensor.size):
			if double_touch_point[i] == 1:
				for j in range(i+1,sensor.size):
					if double_touch_point[j] == 1:
						distance[i][j] += 1
						distance[j][i] += 1
	return np.array(result),distance


def angle_normal(Angle):
	for i in range(Angle.size):
		if Angle[i] < -180.:
			Angle[i] += 360.
		if Angle[i] >  180.:
			Angle[i] -= 360.
	return Angle

def angel_gets_1(ON_Cube,Angle_Cube_2,Angle_Cube_3,Angle_Cube_4,Angle_Cube_5):
	result_Cube   = []
	for time_data in ON_Cube:
		result_Cube.append(np.r_[time_data,
			Angle_Cube_2[int(time_data[3])-1],
			Angle_Cube_3[int(time_data[3])-1],
			Angle_Cube_4[int(time_data[3])-1],
			Angle_Cube_5[int(time_data[3])-1]])
	return np.array(result_Cube)

def angel_gets_2(ON_Cube,Angle_Cube_2,Angle_Cube_3,Angle_Cube_4_1,Angle_Cube_4_2,Angle_Cube_5_1,Angle_Cube_5_2):
	result_Cube   = []
	for time_data in ON_Cube:
		result_Cube.append(np.r_[time_data,
			Angle_Cube_2[int(time_data[3])-1],
			Angle_Cube_3[int(time_data[3])-1],
			Angle_Cube_4_1[int(time_data[3])-1],
			Angle_Cube_4_2[int(time_data[3])-1],
			Angle_Cube_5_1[int(time_data[3])-1],
			Angle_Cube_5_2[int(time_data[3])-1]])
	return np.array(result_Cube)

def angel_gets_3(ON_Cube,Angle_Cube_2,Angle_Cube_3,Angle_Cube_4,Angle_Cube_5):
	result_Cube   = []
	for time_data,t in zip(ON_Cube,range(ON_Cube.shape[0])):
		result_Cube.append(np.r_[time_data,
			Angle_Cube_2[int(time_data[3])-1],
			Angle_Cube_3[int(time_data[3])-1],
			Angle_Cube_4[t],
			Angle_Cube_5[int(time_data[3])-1]])
	return np.array(result_Cube)

def angel_gets_4(ON_Cube,Angle_Cube_2,Angle_Cube_3,Angle_Cube_4,Angle_Cube_5):
	result_Cube   = []
	for time_data,t in zip(ON_Cube,range(ON_Cube.shape[0])):
		result_Cube.append(np.r_[time_data,
			Angle_Cube_2[int(time_data[3])-1],
			Angle_Cube_3[int(time_data[3])-1],
			Angle_Cube_4[int(time_data[3])-1],
			Angle_Cube_5[t]])
	return np.array(result_Cube)

if __name__ == '__main__':

	#触覚センサの位置を決定
	points2        = np.r_[0.,np.sort(create_point(1,1,200-2)),4.]
	points3        = np.r_[0.,np.sort(create_point(1,1,200-2)),4.]
	points4        = np.r_[0.,np.sort(create_point(1,1,200-2)),4.]
	points5        = np.r_[0.,np.sort(create_point(1,1,200-2)),4.]

	#各時刻の各関節角度
	Angle_Cube_2     = np.load("angle_Cube_2.npy")
	Angle_Cube_3     = np.load("angle_Cube_3.npy") 
	Angle_Cube_4_1   = np.load("angle_Cube_4.npy")
	Angle_Cube_4_2   = 180. - np.load("angle_Cube_4.npy")
	Angle_Cube_5_1   = np.load("angle_Cube_5.npy")
	Angle_Cube_5_2   = 180. - np.load("angle_Cube_5.npy")
	
	Angle_2   =  angle_normal(  90.              - Angle_Cube_2.T[0])
	Angle_3   =  angle_normal( 450.              - Angle_Cube_3.T[0])
	Angle_4_1 =  angle_normal( Angle_Cube_2.T[0] - Angle_Cube_4_1.T[0])
	Angle_4_2 =  angle_normal( Angle_Cube_2.T[0] - Angle_Cube_4_2.T[0])
	Angle_5_1 =  angle_normal( Angle_Cube_3.T[0] - Angle_Cube_5_1.T[0])
	Angle_5_2 =  angle_normal( Angle_Cube_3.T[0] - Angle_Cube_5_2.T[0])


	#double touchの発生位置
	ON_Cube_2      = np.load("on_Cube_2.npy")
	ON_Cube_3      = np.load("on_Cube_3.npy")
	ON_Cube_4      = np.load("on_Cube_4.npy")
	ON_Cube_5      = np.load("on_Cube_5.npy")	


	hand_size  = 0.3


	ON_Cube  = np.r_[ON_Cube_2,ON_Cube_3,ON_Cube_4,ON_Cube_5]
	Time     = np.sort(np.unique(ON_Cube.T[3]))
	
	result_Cube_2 =  angel_gets_2(ON_Cube_2,Angle_2,Angle_3,Angle_4_1,Angle_4_2,Angle_5_1,Angle_5_2)
	result_Cube_3 =  angel_gets_2(ON_Cube_3,Angle_2,Angle_3,Angle_4_1,Angle_4_2,Angle_5_1,Angle_5_2)
	result_Cube_4 =  angel_gets_2(ON_Cube_4,Angle_2,Angle_3,Angle_4_1,Angle_4_2,Angle_5_1,Angle_5_2)
	result_Cube_5 =  angel_gets_2(ON_Cube_5,Angle_2,Angle_3,Angle_4_1,Angle_4_2,Angle_5_1,Angle_5_2)
	
	length_2 = np.sqrt(result_Cube_2[:,1]**2 + (result_Cube_2[:,2]+2.)**2 )
	
	length_3 = np.sqrt(result_Cube_3[:,1]**2 + (result_Cube_3[:,2]-2.)**2 )

	length_x = ( 4. * np.sin(result_Cube_4[:,4]*np.pi/180.)      - result_Cube_4[:,1])
	length_y = ((4. * np.cos(result_Cube_4[:,4]*np.pi/180.)- 2.) - result_Cube_4[:,2])
	length_4 = np.sqrt(length_x**2 + length_y**2)

	length_x = ( 4. * np.sin(result_Cube_5[:,5]*np.pi/180.)      - result_Cube_5[:,1])
	length_y = ((4. * np.cos(result_Cube_5[:,5]*np.pi/180.)+ 2.) - result_Cube_5[:,2])
	length_5 = np.sqrt(length_x**2 + length_y**2)


	
	def_Cube_4_1_1 = abs(result_Cube_4[:,1] - \
		(4. * np.sin(result_Cube_4[:,4]*np.pi/180.)   +\
			length_4*np.sin((result_Cube_4[:,6]+result_Cube_4[:,4])*np.pi/180.) ) )
	def_Cube_4_2_1 = abs(result_Cube_4[:,2] - \
		(4. * np.cos(result_Cube_4[:,4]*np.pi/180.)-2.+\
			length_4*np.cos((result_Cube_4[:,6]+result_Cube_4[:,4])*np.pi/180.) ) )


	def_Cube_4_1_2 = abs(result_Cube_4[:,1] - \
		(4. * np.sin(result_Cube_4[:,4]*np.pi/180.)   +\
			length_4*np.sin((result_Cube_4[:,7]+result_Cube_4[:,4])*np.pi/180.) ) )
	def_Cube_4_2_2 = abs(result_Cube_4[:,2] - \
		(4. * np.cos(result_Cube_4[:,4]*np.pi/180.)-2.+\
			length_4*np.cos((result_Cube_4[:,7]+result_Cube_4[:,4])*np.pi/180.) ) )


	def_Cube_4_1  = []
	Angle_4       = []
	for i,j,s,t in zip(def_Cube_4_1_1,def_Cube_4_1_2,result_Cube_4[:,6],result_Cube_4[:,7]):
		if i>j:
			def_Cube_4_1.append(j)
			Angle_4.append(t)
		else:
			def_Cube_4_1.append(i)
			Angle_4.append(s)
	def_Cube_4_1 = np.array(def_Cube_4_1)
	Angle_4      = np.array(Angle_4)


	def_Cube_5_1_1 = abs(result_Cube_5[:,1] - \
		(4. * np.sin(result_Cube_5[:,5]*np.pi/180.)   +\
			length_5*np.sin((result_Cube_5[:,8]+result_Cube_5[:,5])*np.pi/180.) ) )
	def_Cube_5_2_1 = abs(result_Cube_5[:,2] - \
		(4. * np.cos(result_Cube_5[:,5]*np.pi/180.)+2.+\
			length_5*np.cos((result_Cube_5[:,8]+result_Cube_5[:,5])*np.pi/180.) ) )


	def_Cube_5_1_2 = abs(result_Cube_5[:,1] - \
		(4. * np.sin(result_Cube_5[:,5]*np.pi/180.)   +\
			length_5*np.sin((result_Cube_5[:,9]+result_Cube_5[:,5])*np.pi/180.) ) )
	def_Cube_5_2_2 = abs(result_Cube_5[:,2] - \
		(4. * np.cos(result_Cube_5[:,5]*np.pi/180.)+2.+\
			length_5*np.cos((result_Cube_5[:,9]+result_Cube_5[:,5])*np.pi/180.) ) )


	def_Cube_5_1  = []
	Angle_5       = []
	for i,j,s,t in zip(def_Cube_5_1_1,def_Cube_5_1_2,result_Cube_5[:,8],result_Cube_5[:,9]):
		if i>j:
			def_Cube_5_1.append(j)
			Angle_5.append(t)
		else:
			def_Cube_5_1.append(i)
			Angle_5.append(s)
	def_Cube_5_1 = np.array(def_Cube_5_1)
	Angle_5      = np.array(Angle_5)


	result_Cube_2 =  angel_gets_1(ON_Cube_2,Angle_2,Angle_3,Angle_4_2,Angle_5_2)
	result_Cube_3 =  angel_gets_1(ON_Cube_3,Angle_2,Angle_3,Angle_4_2,Angle_5_2)
	result_Cube_4 =  angel_gets_3(ON_Cube_4,Angle_2,Angle_3,Angle_4  ,Angle_5_2)
	result_Cube_5 =  angel_gets_4(ON_Cube_5,Angle_2,Angle_3,Angle_4_2,Angle_5  )
	
	def_Cube_2_1 = abs(result_Cube_2[:,1] -  length_2*np.sin(result_Cube_2[:,4]*np.pi/180.)     )
	def_Cube_2_2 = abs(result_Cube_2[:,2] - (length_2*np.cos(result_Cube_2[:,4]*np.pi/180.)- 2.))	

	def_Cube_3_1 = abs(result_Cube_3[:,1] -  length_3*np.sin(result_Cube_3[:,5]*np.pi/180.)     )
	def_Cube_3_2 = abs(result_Cube_3[:,2] - (length_3*np.cos(result_Cube_3[:,5]*np.pi/180.)+ 2.))


	def_Cube_4_1 = abs(result_Cube_4[:,1] - \
		(4. * np.sin(result_Cube_4[:,4]*np.pi/180.)   +\
			length_4*np.sin((result_Cube_4[:,6]+result_Cube_4[:,4])*np.pi/180.) ) )
	def_Cube_4_2 = abs(result_Cube_4[:,2] - \
		(4. * np.cos(result_Cube_4[:,4]*np.pi/180.)-2.+\
			length_4*np.cos((result_Cube_4[:,6]+result_Cube_4[:,4])*np.pi/180.) ) )

	def_Cube_5_1 = abs(result_Cube_5[:,1] - \
		(4. * np.sin(result_Cube_5[:,5]*np.pi/180.)   +\
			length_5*np.sin((result_Cube_5[:,7]+result_Cube_5[:,5])*np.pi/180.) ) )
	def_Cube_5_2 = abs(result_Cube_5[:,2] - \
		(4. * np.cos(result_Cube_5[:,5]*np.pi/180.)+2.+\
			length_5*np.cos((result_Cube_5[:,7]+result_Cube_5[:,5])*np.pi/180.) ) )


	print(def_Cube_2_1.sum(),def_Cube_2_2.sum())
	print(def_Cube_3_1.sum(),def_Cube_3_2.sum())
	print(def_Cube_4_1.sum(),def_Cube_4_2.sum())
	print(def_Cube_5_1.sum(),def_Cube_5_2.sum())

	result2 ,distance2 = sensor_result(length_2,points2,hand_size)
	result3 ,distance3 = sensor_result(length_3,points3,hand_size)
	result4 ,distance4 = sensor_result(length_4,points4,hand_size)
	result5 ,distance5 = sensor_result(length_5,points5,hand_size)
	
	np.save("point/point2.npy",points2)
	np.save("point/point3.npy",points3)
	np.save("point/point4.npy",points4)
	np.save("point/point5.npy",points5)

	np.save("distance/distance2.npy",distance2)
	np.save("distance/distance3.npy",distance3)
	np.save("distance/distance4.npy",distance4)
	np.save("distance/distance5.npy",distance5)

	np.save("result/result2.npy",result2)
	np.save("result/result3.npy",result3)
	np.save("result/result4.npy",result4)
	np.save("result/result5.npy",result5)

	np.save("result_Cube/result_Cube_2.npy",result_Cube_2)
	np.save("result_Cube/result_Cube_3.npy",result_Cube_3)
	np.save("result_Cube/result_Cube_4.npy",result_Cube_4)
	np.save("result_Cube/result_Cube_5.npy",result_Cube_5)
	
	np.save("length/length_2.npy",length_2)
	np.save("length/length_3.npy",length_3)
	np.save("length/length_4.npy",length_4)
	np.save("length/length_5.npy",length_5)



