# coding: UTF-8

import numpy as np
import scipy as sp

import chainer
from chainer import cuda, Function, gradient_check, Variable 
from chainer import optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

def f_1_x(theta1,theta2,L1,L2,p1,p2):
	x1 = p1 * L1 * np.cos((theta1)*np.pi/180.) + p2 * L2 * np.cos((theta1 + theta2)*np.pi/180.) - (4.0 / 2.0)
	return x1

def f_1_y(theta1,theta2,L1,L2,p1,p2):
	y1 = p1 * L1 * np.sin((theta1)*np.pi/180.) + p2 * L2 * np.sin((theta1 + theta2)*np.pi/180.)
	return y1

def f_2_x(theta3,theta4,L3,L4,p3,p4):
	x2 = p3 * L3 * np.cos((theta3)*np.pi/180.) + p4 * L4 * np.cos((theta3 + theta4)*np.pi/180.) + (4.0 / 2.0)
	return x2

def f_2_y(theta3,theta4,L3,L4,p3,p4):
	y2 = p3 * L3 * np.sin((theta3)*np.pi/180.) + p4 * L4 * np.sin((theta3 + theta4)*np.pi/180.)
	return y2



if __name__ == '__main__':
	Cube_2          = np.load("Cube_2.npy")
	Cube_3          = np.load("Cube_3.npy")
	Cube_4          = np.load("Cube_4.npy")
	Cube_5          = np.load("Cube_5.npy")
	time            = np.sort(np.unique(np.r_[Cube_2.T[3],Cube_3.T[3],Cube_4.T[3],Cube_5.T[3]]) )

	arm_2           = np.load("arm_2.npy")
	arm_3           = np.load("arm_3.npy")
	arm_4           = np.load("arm_4.npy")
	arm_5           = np.load("arm_5.npy")

	result_2        = np.load("result_2.npy")
	result_3        = np.load("result_3.npy")
	result_4        = np.load("result_4.npy")
	result_5        = np.load("result_5.npy")

	estimate_point2 = (1.-np.load("./../esimate_point/estimate_point2.npy"))*4
	estimate_point3 = (1.-np.load("./../esimate_point/estimate_point3.npy"))*4
	estimate_point4 = (   np.load("./../esimate_point/estimate_point4.npy"))*4
	estimate_point5 = (   np.load("./../esimate_point/estimate_point5.npy"))*4

	print(abs(arm_2-result_2.dot(estimate_point2)/result_2.sum(1)).mean())
	print(abs(arm_3-result_3.dot(estimate_point3)/result_3.sum(1)).mean())
	print(abs(arm_4-result_4.dot(estimate_point4)/result_4.sum(1)).mean())
	print(abs(arm_5-result_5.dot(estimate_point5)/result_5.sum(1)).mean())

	estimate_length2 = result_2.dot(estimate_point2)/result_2.sum(1)
	estimate_length3 = result_3.dot(estimate_point3)/result_3.sum(1)
	estimate_length4 = result_4.dot(estimate_point4)/result_4.sum(1)
	estimate_length5 = result_5.dot(estimate_point5)/result_5.sum(1)

	ct2              = 0
	ct3              = 0
	ct4              = 0
	ct5              = 0

	Cube_2_x_diff       = np.diff(Cube_2.T[1])
	Cube_2_y_diff       = np.diff(Cube_2.T[2])
	Cube_2_time_diff    = np.diff(Cube_2.T[3])
	Cube_2_angle_1_diff = np.diff(Cube_2.T[4])
	Cube_2_angle_2_diff = np.diff(Cube_2.T[6])

	Cube_3_x_diff       = np.diff(Cube_3.T[1])
	Cube_3_y_diff       = np.diff(Cube_3.T[2])
	Cube_3_time_diff    = np.diff(Cube_3.T[3])
	Cube_3_angle_1_diff = np.diff(Cube_3.T[5])
	Cube_3_angle_2_diff = np.diff(Cube_3.T[7])

	Cube_4_x_diff       = np.diff(Cube_4.T[1])
	Cube_4_y_diff       = np.diff(Cube_4.T[2])
	Cube_4_time_diff    = np.diff(Cube_4.T[3])
	Cube_4_angle_1_diff = np.diff(Cube_4.T[4])
	Cube_4_angle_2_diff = np.diff(Cube_4.T[6])

	Cube_5_x_diff       = np.diff(Cube_5.T[1])
	Cube_5_y_diff       = np.diff(Cube_5.T[2])
	Cube_5_time_diff    = np.diff(Cube_5.T[3])
	Cube_5_angle_1_diff = np.diff(Cube_5.T[5])
	Cube_5_angle_2_diff = np.diff(Cube_5.T[7])

	diff2_1 = Cube_2.T[2] - f_1_x(Cube_2.T[4],Cube_2.T[6],1.0,1.0,arm_2,np.zeros(arm_2.shape[0]))
	diff2_2 = Cube_2.T[1] - f_1_y(Cube_2.T[4],Cube_2.T[6],1.0,1.0,arm_2,np.zeros(arm_2.shape[0]))
	
	diff3_1 = Cube_3.T[2] - f_2_x(Cube_3.T[5],Cube_3.T[7],1.0,1.0,arm_3,np.zeros(arm_3.shape[0]))
	diff3_2 = Cube_3.T[1] - f_2_y(Cube_3.T[5],Cube_3.T[7],1.0,1.0,arm_3,np.zeros(arm_3.shape[0]))
	
	diff4_1 = Cube_4.T[2] - f_1_x(Cube_4.T[4],Cube_4.T[6],1.0,1.0,np.ones(arm_4.shape[0])*4.,arm_4)
	diff4_2 = Cube_4.T[1] - f_1_y(Cube_4.T[4],Cube_4.T[6],1.0,1.0,np.ones(arm_4.shape[0])*4.,arm_4)
	
	diff5_1 = Cube_5.T[2] - f_2_x(Cube_5.T[5],Cube_5.T[7],1.0,1.0,np.ones(arm_5.shape[0])*4.,arm_5)
	diff5_2 = Cube_5.T[1] - f_2_y(Cube_5.T[5],Cube_5.T[7],1.0,1.0,np.ones(arm_5.shape[0])*4.,arm_5)
	
	print(abs(diff2_1).mean(),abs(diff2_2).mean())
	print(abs(diff3_1).mean(),abs(diff3_2).mean())
	print(abs(diff4_1).mean(),abs(diff4_2).mean())
	print(abs(diff5_1).mean(),abs(diff5_2).mean())
'''
	for i in  time:
		print(int(i),ct2,ct3,ct4,ct5)
		if Cube_2[ct2,3] == i:
			if Cube_2[ct2,3] != Cube_2.T[3].max():
				ct2 += 1

		if Cube_3[ct3,3] == i:
			if Cube_3[ct3,3] != Cube_3.T[3].max():
				ct3 += 1

		if Cube_4[ct4,3] == i:
			if Cube_4[ct4,3] != Cube_4.T[3].max():
				ct4 += 1

		if Cube_5[ct5,3] == i:
			if Cube_5[ct5,3] != Cube_5.T[3].max():
				ct5 += 1
'''


