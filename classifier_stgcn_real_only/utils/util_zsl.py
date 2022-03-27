import numpy as np


def unit_vector(vector):
	return vector / np.linalg.norm(vector)

def angle_between(v1,v2):
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def area_triangle(v1,v2,v3):
	a = np.linalg.norm(v1 - v2)
	b = np.linalg.norm(v2 - v3)
	c = np.linalg.norm(v3 - v1)
	s = (a+b+c) / 2.0
	return np.sqrt(s * (s-a) * (s-b) * (s-c))

def distance_between(v1,v2):
	return np.linalg.norm(v1-v2)