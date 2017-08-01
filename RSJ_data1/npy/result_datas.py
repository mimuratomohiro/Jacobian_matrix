# coding: UTF-8

import numpy as np
import scipy as sp

if __name__ == '__main__':
	result_Cube_2 = np.load("result_Cube/result_Cube_2.npy")
	result_Cube_3 = np.load("result_Cube/result_Cube_3.npy")
	result_Cube_4 = np.load("result_Cube/result_Cube_4.npy")
	result_Cube_5 = np.load("result_Cube/result_Cube_5.npy")

	result2       = np.load("result/result2.npy")
	result3       = np.load("result/result3.npy")
	result4       = np.load("result/result4.npy")
	result5       = np.load("result/result5.npy")

	length_2      =	np.load("length/length_2.npy")
	length_3      =	np.load("length/length_3.npy")
	length_4      =	np.load("length/length_4.npy")
	length_5      =	np.load("length/length_5.npy")

	time_Cube_2   = np.unique(result_Cube_2.T[3])
	time_Cube_3   = np.unique(result_Cube_3.T[3])
	time_Cube_4   = np.unique(result_Cube_4.T[3])
	time_Cube_5   = np.unique(result_Cube_5.T[3])
	time_Cube_all = np.sort(np.unique(np.r_[time_Cube_2,time_Cube_3,time_Cube_4,time_Cube_5]))

	
	ct2           = 0
	ct3           = 0
	ct4           = 0
	ct5           = 0

	time_joint_all  = []
	time_all        = []

	for i in time_Cube_all:
		next_frag   = np.ones(4)
		time_joint  = np.zeros(4)

		while next_frag.sum() != 0.:
			if ct2 == result_Cube_2.shape[0]:
				next_frag[0] = 0
			elif result_Cube_2[ct2,3] == i:
				ct2  += 1
				time_joint[0] = 1
			else:
				next_frag[0] = 0


			if ct3 == result_Cube_3.shape[0]:
				next_frag[1] = 0
			elif result_Cube_3[ct3,3] == i:
				ct3  += 1
				time_joint[1] = 1
			else:
				next_frag[1] = 0


			if ct4 == result_Cube_4.shape[0]:
				next_frag[2] = 0
			elif result_Cube_4[ct4,3] == i:
				ct4  += 1
				time_joint[2] = 1
			else:
				next_frag[2] = 0


			if ct5 == result_Cube_5.shape[0]:
				next_frag[3] = 0
			elif result_Cube_5[ct5,3] == i:
				ct5  += 1
				time_joint[3] = 1
			else:
				next_frag[3] = 0

		if time_joint.sum() == 2.:
			time_joint_all.append(time_joint)
			time_all.append(i)

	time_joint_all = np.array(time_joint_all)
	time_all       = np.array(time_all)

	ct2            = 0
	ct3            = 0
	ct4            = 0
	ct5            = 0

	Cube_2         = []
	Cube_3         = []
	Cube_4         = []
	Cube_5         = []

	arm_2          = []
	arm_3          = []
	arm_4          = []
	arm_5          = []

	result_2       = []
	result_3       = []
	result_4       = []
	result_5       = []

	for i in time_all:
		next_frag  = np.ones(5)
		while next_frag.sum() != 0.:
			if ct2 == result_Cube_2.shape[0]:
				next_frag[0] = 0
			elif result_Cube_2[ct2,3] == i:
				if next_frag[4] == 1:
					Cube_2.append(result_Cube_2[ct2])
					arm_2.append(length_2[ct2])
					result_2.append(result2[ct2])
				ct2  += 1
			elif result_Cube_2[ct2,3] < i:
				ct2  += 1
			else:
				next_frag[0] = 0


			if ct3 == result_Cube_3.shape[0]:
				next_frag[1] = 0
			elif result_Cube_3[ct3,3] == i:
				if next_frag[4] == 1:
					Cube_3.append(result_Cube_3[ct3])
					arm_3.append(length_3[ct3])
					result_3.append(result3[ct3])
				ct3  += 1
			elif result_Cube_3[ct3,3] < i:
				ct3  += 1
			else:
				next_frag[1] = 0


			if ct4 == result_Cube_4.shape[0]:
				next_frag[2] = 0
			elif result_Cube_4[ct4,3] == i:
				if next_frag[4] == 1:
					Cube_4.append(result_Cube_4[ct4])
					arm_4.append(length_4[ct4])
					result_4.append(result4[ct4])
				ct4  += 1
			elif result_Cube_4[ct4,3] < i:
				ct4  += 1
			else:
				next_frag[2] = 0


			if ct5 == result_Cube_5.shape[0]:
				next_frag[3] = 0
			elif result_Cube_5[ct5,3] == i:
				if next_frag[4] == 1:
					Cube_5.append(result_Cube_5[ct5])
					arm_5.append(length_5[ct5])
					result_5.append(result5[ct5])
				ct5  += 1
			elif result_Cube_5[ct5,3] < i:
				ct5  += 1
			else:
				next_frag[3] = 0

			if next_frag[4] == 1:
				next_frag[4] = 0

	Cube_2   = np.array(Cube_2)
	Cube_3   = np.array(Cube_3)
	Cube_4   = np.array(Cube_4)
	Cube_5   = np.array(Cube_5)

	arm_2    = np.array(arm_2)
	arm_3    = np.array(arm_3)
	arm_4    = np.array(arm_4)
	arm_5    = np.array(arm_5)

	result_2 = np.array(result_2)
	result_3 = np.array(result_3)
	result_4 = np.array(result_4)
	result_5 = np.array(result_5)

	np.save("data/Cube_2.npy",Cube_2)
	np.save("data/Cube_3.npy",Cube_3)
	np.save("data/Cube_4.npy",Cube_4)
	np.save("data/Cube_5.npy",Cube_5)

	np.save("data/arm_2.npy",arm_2)
	np.save("data/arm_3.npy",arm_3)
	np.save("data/arm_4.npy",arm_4)
	np.save("data/arm_5.npy",arm_5)

	np.save("data/result_2.npy",result_2)
	np.save("data/result_3.npy",result_3)
	np.save("data/result_4.npy",result_4)
	np.save("data/result_5.npy",result_5)

	

