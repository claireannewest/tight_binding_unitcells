import numpy as np
import matplotlib.pyplot as plt
import yaml
open_param_file = open('parameter_files/parameters.yaml')
param = yaml.load(open_param_file)
numparticles = 1
filename='dimer'
coordinates = np.loadtxt('parameter_files/coords_'+str(filename)+'.txt')
base_vectors = np.loadtxt('parameter_files/basevecs_'+str(filename)+'.txt')
numRhombs = len(coordinates)
coordinates = np.concatenate((coordinates, coordinates),axis=0)
### corners is organized: long axis rhomb 1, long axis rhomb 2 , ... , short axis rhomb 1, shortaxis rhomb 2, ...
corners = base_vectors+coordinates

def plot_og(which):
	plt.scatter(coordinates[:,0], coordinates[:,1], color='black')
	plt.scatter(corners[:,0], corners[:,1], color='red')
	plt.scatter(coordinates[which,0], coordinates[which,1], color='green')
	plt.scatter(corners[which,0], corners[which,1], color='green')
	plt.show()

def find_allcorners(): ### find all corners ### 
	new_corners = np.zeros((int(2*len(corners)), 2))
	dupl_coords = np.zeros((int(2*len(corners)), 2))

	for i in range(0, 2*numRhombs):
		length = 2*np.linalg.norm(corners[i,:]-coordinates[i,:])
		theta = np.arctan2(coordinates[i,1]-corners[i,1],coordinates[i,0]-corners[i,0] )
		#print(theta)
		new_corners[2*i, :] = corners[i,:]
		new_corners[2*i+1,0] = length*np.cos(theta)+corners[i,0]
		new_corners[2*i+1,1] = length*np.sin(theta)+corners[i,1]

		dupl_coords[2*i, :] = coordinates[i,:]
		dupl_coords[2*i+1, :] = coordinates[i,:]
	return new_corners, dupl_coords 

def plot_allcorners(which):
	new_corners, dupl_coords = find_allcorners()
	plt.scatter(coordinates[:,0], coordinates[:,1], s=100,color='black')
	plt.scatter(new_corners[:,0], new_corners[:,1], color='red')
	plt.scatter(dupl_coords[which,0], dupl_coords[which,1], color='green')
	plt.scatter(new_corners[which,0], new_corners[which,1], color='green')
	plt.quiver(new_corners[which,0], new_corners[which,1], -new_corners[which,0]+new_corners[which+1,0], -new_corners[which,1]+ new_corners[which+1,1], scale=1 )
	plt.axis('equal')
	plt.show()

def find_shrunk_corners(dl=62, ds=70.5):
	### Originally, Xuan's code made:
	### side length = 200 nm, long axis diameter = 324 nm, short axis diameter = 235 nm
	### But to fit modes, I'll pull in the corners by dl and ds
	corners, rhomb_cent = find_allcorners()
	theta = np.arctan2(rhomb_cent[:,1]-corners[:,1],rhomb_cent[:,0]-corners[:,0])
	shrunk_x = np.zeros((len(theta))); shrunk_y = np.zeros((len(theta)))

	shrunk_x[0:2*numRhombs] = dl*np.cos(theta[0:2*numRhombs])+corners[0:2*numRhombs,0]
	shrunk_y[0:2*numRhombs] = dl*np.sin(theta[0:2*numRhombs])+corners[0:2*numRhombs,1]
	shrunk_x[2*numRhombs:] = ds*np.cos(theta[2*numRhombs:])+corners[2*numRhombs:,0]
	shrunk_y[2*numRhombs:] = ds*np.sin(theta[2*numRhombs:])+corners[2*numRhombs:,1]

	shrunk = np.column_stack((shrunk_x, shrunk_y))
	return corners, rhomb_cent, shrunk, theta

def rotate_normalmodes():
	og_corners, rhomb_cent, shrunk, theta = find_shrunk_corners()
	rotate_l = np.zeros((len(shrunk), 2))
	rotate_s = np.zeros((len(shrunk), 2))
	rotate_q = np.zeros((len(shrunk), 2))
	fix_rotate_q = np.zeros((len(shrunk), 2))

	mode_l = np.loadtxt('../tb_quasicrystal_4sph/output_files/normal_mode_0.txt',skiprows=1)*1E7
	mode_s = np.loadtxt('../tb_quasicrystal_4sph/output_files/normal_mode_1.txt',skiprows=1)*1E7
	mode_q = np.loadtxt('../tb_quasicrystal_4sph/output_files/normal_mode_2.txt',skiprows=1)*1E7

	for rhombi in range(0, numRhombs):
		xdiff = shrunk[2*rhombi+1,0]-shrunk[2*rhombi,0]
		ydiff = shrunk[2*rhombi+1,1]-shrunk[2*rhombi,1]
		theta = np.pi-np.arctan2(xdiff, ydiff)
	#############################################
		## L ##
		original = mode_l[3,2:4]
		rotate_l[int(2*rhombi),0] = original[0]*np.cos(theta) - original[1]*np.sin(theta)
		rotate_l[int(2*rhombi),1] = original[0]*np.sin(theta) + original[1]*np.cos(theta)

		original = mode_l[2,2:4]
		rotate_l[int(2*rhombi)+1,0] = original[0]*np.cos(theta) - original[1]*np.sin(theta)
		rotate_l[int(2*rhombi)+1,1] = original[0]*np.sin(theta) + original[1]*np.cos(theta)

		original = mode_l[0,2:4]
		rotate_l[int(2*rhombi+2*numRhombs),0] = original[0]*np.cos(theta) - original[1]*np.sin(theta)
		rotate_l[int(2*rhombi+2*numRhombs),1] = original[0]*np.sin(theta) + original[1]*np.cos(theta)

		original = mode_l[1,2:4]
		rotate_l[int(2*rhombi+2*numRhombs+1),0] = original[0]*np.cos(theta) - original[1]*np.sin(theta)
		rotate_l[int(2*rhombi+2*numRhombs+1),1] = original[0]*np.sin(theta) + original[1]*np.cos(theta)

		## S ##
		original = mode_s[3,2:4]
		rotate_s[int(2*rhombi),0] = original[0]*np.cos(theta) - original[1]*np.sin(theta)
		rotate_s[int(2*rhombi),1] = original[0]*np.sin(theta) + original[1]*np.cos(theta)
		original = mode_s[2,2:4]
		rotate_s[int(2*rhombi)+1,0] = original[0]*np.cos(theta) - original[1]*np.sin(theta)
		rotate_s[int(2*rhombi)+1,1] = original[0]*np.sin(theta) + original[1]*np.cos(theta)
		original = mode_s[0,2:4]
		rotate_s[int(2*rhombi+2*numRhombs),0] = original[0]*np.cos(theta) - original[1]*np.sin(theta)
		rotate_s[int(2*rhombi+2*numRhombs),1] = original[0]*np.sin(theta) + original[1]*np.cos(theta)
		original = mode_s[1,2:4]
		rotate_s[int(2*rhombi+2*numRhombs+1),0] = original[0]*np.cos(theta) - original[1]*np.sin(theta)
		rotate_s[int(2*rhombi+2*numRhombs+1),1] = original[0]*np.sin(theta) + original[1]*np.cos(theta)

		## Q ##
		original = mode_q[3,2:4]
		rotate_q[int(2*rhombi),0] = original[0]*np.cos(theta) - original[1]*np.sin(theta)
		rotate_q[int(2*rhombi),1] = original[0]*np.sin(theta) + original[1]*np.cos(theta)
		original = mode_q[2,2:4]
		rotate_q[int(2*rhombi)+1,0] = original[0]*np.cos(theta) - original[1]*np.sin(theta)
		rotate_q[int(2*rhombi)+1,1] = original[0]*np.sin(theta) + original[1]*np.cos(theta)
		original = mode_q[0,2:4]
		rotate_q[int(2*rhombi+2*numRhombs),0] = original[0]*np.cos(theta) - original[1]*np.sin(theta)
		rotate_q[int(2*rhombi+2*numRhombs),1] = original[0]*np.sin(theta) + original[1]*np.cos(theta)
		original = mode_q[1,2:4]
		rotate_q[int(2*rhombi+2*numRhombs+1),0] = original[0]*np.cos(theta) - original[1]*np.sin(theta)
		rotate_q[int(2*rhombi+2*numRhombs+1),1] = original[0]*np.sin(theta) + original[1]*np.cos(theta)

	fix_rotate_q = rotate_q # this fixes the problem that I don't know what corner I'm grabbing. It only affects Q mode
	for rhombi in range(0, numRhombs):
		if np.cross(rotate_q[int(2*rhombi+2*numRhombs),:]+shrunk[int(2*rhombi+2*numRhombs),:], rotate_q[int(2*rhombi),:]+shrunk[int(2*rhombi),:]) <0:
			fix_rotate_q[int(2*rhombi+2*numRhombs),:] = rotate_q[int(2*rhombi+2*numRhombs+1),:]
			fix_rotate_q[int(2*rhombi+2*numRhombs+1),:] = -rotate_q[int(2*rhombi+2*numRhombs),:]

	return og_corners, rhomb_cent, shrunk, rotate_l, rotate_s, fix_rotate_q


def plot_shrunk(which_sph, which_mode):
	og_corners, rhomb_cent, shrunk, rotate_l, rotate_s, rotate_q = rotate_normalmodes()
	fig = plt.figure(1, figsize=[5.,5.])
	ax = plt.gca()
	ax.set_aspect('equal', adjustable='box')
	plt.scatter(rhomb_cent[:,0], rhomb_cent[:,1], color='black')
	plt.scatter(og_corners[:,0], og_corners[:,1], color='black')
	plt.scatter(shrunk[:,0], shrunk[:,1], color='blue')
	if which_mode == 'l': plt.quiver(shrunk[:,0], shrunk[:,1], rotate_l[:,0], rotate_l[:,1],scale=0.9)
	if which_mode == 's': plt.quiver(shrunk[:,0], shrunk[:,1], rotate_s[:,0], rotate_s[:,1],scale=2.3)
	if which_mode == 'q': plt.quiver(shrunk[:,0], shrunk[:,1], rotate_q[:,0], rotate_q[:,1],scale=2.4)
	plt.scatter(rhomb_cent[which_sph,0], rhomb_cent[which_sph,1], color='green')
	plt.scatter(og_corners[which_sph,0], og_corners[which_sph,1], color='green')
	plt.scatter(shrunk[which_sph,0], shrunk[which_sph,1], color='green')
	#print('dist from original corner', np.linalg.norm(og_corners[which_sph]-shrunk[which_sph]))
	plt.show()

def final_writing():
	og_corners, rhomb_cent, shrunk, rotate_l, rotate_s, rotate_q = rotate_normalmodes()
	final_write = np.column_stack((rhomb_cent, shrunk, rotate_l, rotate_s, rotate_q))

	### Writing time ###
	file = open(str('inputs_')+str(filename)+str('_0403.txt'),'w')
	file.write( 'Rhomb Center [nm]' + '\t' + '\t' + 'Dip Center [nm]' +  '\t' +  '\t' +  '\t' +'L [nm]' + '\t' +'S [nm]' + '\t' +'Q [nm]' + '\t' + '\n')
	for j in range(0, len(final_write)):
		file.write("%.3f" % final_write[j,:][0] + '\t' + "%.3f" % final_write[j,:][1] + '\t' +
				   "%.3f" % final_write[j,:][2] + '\t' + "%.3f" % final_write[j,:][3] + '\t' + 
				   "%.5f" % final_write[j,:][4] + '\t' + "%.5f" % final_write[j,:][5] + '\t' + 
				   "%.5f" % final_write[j,:][6] + '\t' + "%.5f" % final_write[j,:][7] + '\t' + 
				   "%.5f" % final_write[j,:][8] + '\t' + "%.5f" % final_write[j,:][9] + '\n')
	file.close()

final_writing()
