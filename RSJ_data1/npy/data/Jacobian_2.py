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
	x1 = p1*L1*np.cos((theta1)*np.pi/180.)+p2*L2*np.cos((theta1+theta2)*np.pi/180.)-(4./2.)
	return x1

def f_1_y(theta1,theta2,L1,L2,p1,p2):
	y1 = p1*L1*np.sin((theta1)*np.pi/180.)+p2*L2*np.sin((theta1+theta2)*np.pi/180.)
	return y1

def f_2_x(theta3,theta4,L3,L4,p3,p4):
	x2 = p3*L3*np.cos((theta3)*np.pi/180.)+p4*L4*np.cos((theta3+theta4)*np.pi/180.)+(4./2.)
	return x2

def f_2_y(theta3,theta4,L3,L4,p3,p4):
	y2 = p3*L3*np.sin((theta3)*np.pi/180.)+p4*L4*np.sin((theta3+theta4)*np.pi/180.)
	return y2

def f_1_x_theta1(theta1,theta2,L1,L2,p1,p2):
	x1 = -p1*L1*np.sin((theta1)*np.pi/180.)-p2*L2*np.sin((theta1+theta2)*np.pi/180.)
	return x1

def f_1_x_theta2(theta1,theta2,L1,L2,p1,p2):
	x1 = -p2*L2*np.sin((theta1+theta2)*np.pi/180.) 
	return x1

def f_1_y_theta1(theta1,theta2,L1,L2,p1,p2):
	y1 =  p1*L1*np.cos((theta1)*np.pi/180.)+p2*L2*np.cos((theta1+theta2)*np.pi/180.)
	return y1

def f_1_y_theta2(theta1,theta2,L1,L2,p1,p2):
	y1 =  p2*L2*np.cos((theta1+theta2)*np.pi/180.)
	return y1

def f_2_x_theta3(theta3,theta4,L3,L4,p3,p4):
	x2 = -p3*L3*np.sin((theta3)*np.pi/180.)-p4*L4*np.sin((theta3+theta4)*np.pi/180.) 
	return x2

def f_2_x_theta4(theta3,theta4,L3,L4,p3,p4):
	x2 = -p4*L4*np.sin((theta3+theta4)*np.pi/180.) 
	return x2

def f_2_y_theta3(theta3,theta4,L3,L4,p3,p4):
	y2 =  p3*L3*np.cos((theta3)*np.pi/180.)+p4*L4*np.cos((theta3+theta4)*np.pi/180.)
	return y2

def f_2_y_theta4(theta3,theta4,L3,L4,p3,p4):
	y2 =  p4*L4*np.cos((theta3+theta4)*np.pi/180.)
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
	
	estimate_length2 = result_2.dot(estimate_point2)/result_2.sum(1)
	estimate_length3 = result_3.dot(estimate_point3)/result_3.sum(1)
	estimate_length4 = result_4.dot(estimate_point4)/result_4.sum(1)
	estimate_length5 = result_5.dot(estimate_point5)/result_5.sum(1)

######################################################################################
	Cube_x_diff       = np.diff(Cube_2.T[1])
	Cube_y_diff       = np.diff(Cube_2.T[2])
	Cube_time_diff    = np.diff(Cube_2.T[3])
	Cube_angle_1_diff = np.diff(Cube_2.T[4])
	Cube_angle_2_diff = np.diff(Cube_2.T[6])


	variable_1        = Cube_2.T[4]
	variable_2        = Cube_2.T[6]
	variable_3        = np.ones(arm_2.shape[0])*4.
	variable_4        = np.ones(arm_2.shape[0])*4.
	variable_5        = estimate_length2
	variable_6        = np.zeros(arm_2.shape[0])

	Diff_1            = f_1_x(variable_1,variable_2,variable_3,variable_4,variable_5,variable_6)
	Diff_1_theta1     = f_1_x_theta1(variable_1,variable_2,variable_3,variable_4,variable_5,variable_6)
	Diff_1_theta2     = f_1_x_theta2(variable_1,variable_2,variable_3,variable_4,variable_5,variable_6)
	Diff_2            = f_1_y(variable_1,variable_2,variable_3,variable_4,variable_5,variable_6)
	Diff_2_theta1     = f_1_y_theta1(variable_1,variable_2,variable_3,variable_4,variable_5,variable_6)
	Diff_2_theta2     = f_1_y_theta2(variable_1,variable_2,variable_3,variable_4,variable_5,variable_6)
	Jacobian_1        = Cube_angle_1_diff*np.pi/180. * Diff_1_theta1[:-1] + Cube_angle_2_diff*np.pi/180. * Diff_1_theta2[:-1] 
	Jacobian_2        = Cube_angle_1_diff*np.pi/180. * Diff_2_theta1[:-1] + Cube_angle_2_diff*np.pi/180. * Diff_2_theta2[:-1] 
	
	sum_errer         = 0.
	for i,j,t in zip((np.diff(Diff_2) - Jacobian_2)**2,(np.diff(Diff_1) - Jacobian_1)**2,Cube_time_diff):
		if t == 1.:
			sum_errer  += i + j
	print(sum_errer)


######################################################################################
	Cube_x_diff       = np.diff(Cube_3.T[1])
	Cube_y_diff       = np.diff(Cube_3.T[2])
	Cube_time_diff    = np.diff(Cube_3.T[3])
	Cube_angle_1_diff = np.diff(Cube_3.T[5])
	Cube_angle_2_diff = np.diff(Cube_3.T[7])

	variable_1        = Cube_3.T[5]
	variable_2        = Cube_3.T[7]
	variable_3        = np.ones(arm_3.shape[0])*4.
	variable_4        = np.ones(arm_3.shape[0])*4.
	variable_5        = estimate_length3
	variable_6        = np.zeros(arm_3.shape[0])

	Diff_1            = f_2_x(variable_1,variable_2,variable_3,variable_4,variable_5,variable_6)
	Diff_1_theta1     = f_2_x_theta3(variable_1,variable_2,variable_3,variable_4,variable_5,variable_6)
	Diff_1_theta2     = f_2_x_theta4(variable_1,variable_2,variable_3,variable_4,variable_5,variable_6)
	Diff_2            = f_2_y(variable_1,variable_2,variable_3,variable_4,variable_5,variable_6)
	Diff_2_theta1     = f_2_y_theta3(variable_1,variable_2,variable_3,variable_4,variable_5,variable_6)
	Diff_2_theta2     = f_2_y_theta4(variable_1,variable_2,variable_3,variable_4,variable_5,variable_6)
	Jacobian_1        = Cube_angle_1_diff*np.pi/180. * Diff_1_theta1[:-1] + Cube_angle_2_diff*np.pi/180. * Diff_1_theta2[:-1] 
	Jacobian_2        = Cube_angle_1_diff*np.pi/180. * Diff_2_theta1[:-1] + Cube_angle_2_diff*np.pi/180. * Diff_2_theta2[:-1] 
	
	sum_errer         = 0.
	for i,j,t in zip((np.diff(Diff_2) - Jacobian_2)**2,(np.diff(Diff_1) - Jacobian_1)**2,Cube_time_diff):
		if t == 1.:
			sum_errer  += i + j
	print(sum_errer)

######################################################################################
	Cube_x_diff       = np.diff(Cube_4.T[1])
	Cube_y_diff       = np.diff(Cube_4.T[2])
	Cube_time_diff    = np.diff(Cube_4.T[3])
	Cube_angle_1_diff = np.diff(Cube_4.T[4])
	Cube_angle_2_diff = np.diff(Cube_4.T[6])


	variable_1        = Cube_4.T[4]
	variable_2        = Cube_4.T[6]
	variable_3        = np.ones(arm_4.shape[0])*4.
	variable_4        = np.ones(arm_4.shape[0])*4.
	variable_5        = np.ones(arm_4.shape[0])
	variable_6        = estimate_length4

	Diff_1            = f_1_x(variable_1,variable_2,variable_3,variable_4,variable_5,variable_6)
	Diff_1_theta1     = f_1_x_theta1(variable_1,variable_2,variable_3,variable_4,variable_5,variable_6)
	Diff_1_theta2     = f_1_x_theta2(variable_1,variable_2,variable_3,variable_4,variable_5,variable_6)
	Diff_2            = f_1_y(variable_1,variable_2,variable_3,variable_4,variable_5,variable_6)
	Diff_2_theta1     = f_1_y_theta1(variable_1,variable_2,variable_3,variable_4,variable_5,variable_6)
	Diff_2_theta2     = f_1_y_theta2(variable_1,variable_2,variable_3,variable_4,variable_5,variable_6)
	Jacobian_1        = Cube_angle_1_diff*np.pi/180. * Diff_1_theta1[:-1] + Cube_angle_2_diff*np.pi/180. * Diff_1_theta2[:-1] 
	Jacobian_2        = Cube_angle_1_diff*np.pi/180. * Diff_2_theta1[:-1] + Cube_angle_2_diff*np.pi/180. * Diff_2_theta2[:-1] 
	
	sum_errer         = 0.
	for i,j,t in zip((np.diff(Diff_2) - Jacobian_2)**2,(np.diff(Diff_1) - Jacobian_1)**2,Cube_time_diff):
		if t == 1.:
			sum_errer  += i + j
	print(sum_errer)


######################################################################################
	Cube_x_diff       = np.diff(Cube_5.T[1])
	Cube_y_diff       = np.diff(Cube_5.T[2])
	Cube_time_diff    = np.diff(Cube_5.T[3])
	Cube_angle_1_diff = np.diff(Cube_5.T[5])
	Cube_angle_2_diff = np.diff(Cube_5.T[7])

	variable_1        = Cube_5.T[5]
	variable_2        = Cube_5.T[7]
	variable_3        = np.ones(arm_5.shape[0])*4.
	variable_4        = np.ones(arm_5.shape[0])*4.
	variable_5        = np.ones(arm_5.shape[0])
	variable_6        = estimate_length5

	Diff_1            = f_2_x(variable_1,variable_2,variable_3,variable_4,variable_5,variable_6)
	Diff_1_theta1     = f_2_x_theta3(variable_1,variable_2,variable_3,variable_4,variable_5,variable_6)
	Diff_1_theta2     = f_2_x_theta4(variable_1,variable_2,variable_3,variable_4,variable_5,variable_6)
	Diff_2            = f_2_y(variable_1,variable_2,variable_3,variable_4,variable_5,variable_6)
	Diff_2_theta1     = f_2_y_theta3(variable_1,variable_2,variable_3,variable_4,variable_5,variable_6)
	Diff_2_theta2     = f_2_y_theta4(variable_1,variable_2,variable_3,variable_4,variable_5,variable_6)
	Jacobian_1        = Cube_angle_1_diff*np.pi/180. * Diff_1_theta1[:-1] + Cube_angle_2_diff*np.pi/180. * Diff_1_theta2[:-1] 
	Jacobian_2        = Cube_angle_1_diff*np.pi/180. * Diff_2_theta1[:-1] + Cube_angle_2_diff*np.pi/180. * Diff_2_theta2[:-1] 
	
	sum_errer         = 0.
	for i,j,t in zip((np.diff(Diff_2) - Jacobian_2)**2,(np.diff(Diff_1) - Jacobian_1)**2,Cube_time_diff):
		if t == 1.:
			sum_errer  += i + j
	print(sum_errer)
	#for i,j in zip(Cube.T[2] - Diff_1,Cube.T[1] - Diff_2):
	#	print(i,j)


