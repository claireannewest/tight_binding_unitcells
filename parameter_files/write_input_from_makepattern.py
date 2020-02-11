import numpy as np
import matplotlib.pyplot as plt

filename = 'trimertogether'

coordinates = np.loadtxt(str('parameter_files/coords_')+str(filename)+str('.txt')) #outputs of make_pattern, the (x,y) of the center of each rhombus
base_vectors = np.loadtxt(str('parameter_files/basevecs_')+str(filename)+str('.txt')) #outputs of make_pattern
coords_concat = coordinates
coords_concat = np.concatenate((coords_concat, coordinates),axis=0) # concat to have same dimentions as corners
corners_x = base_vectors[:,0]+coords_concat[:,0] # the x coord of 2 of the 4 corners of each rhombus
corners_y = base_vectors[:,1]+coords_concat[:,1] # the y coord of 2 of the 4 corners of each rhombus

numRhombs = len(coords_concat)/2
numIndModes = 3

### This code will take the outputs of the make_pattern code outputs, namely 
### the coords_concat of each rhombus, and two of the four corners. Then, it will 
### make an input file where the number of: 
### Number of rows = (number of rhombuses)*(number of independent modes)
### 			   = (number of rhombuses)*(3) --> because D_l, D_s, Q
### Column 0, 1 = (x, y) coords of rhombus
### Column 2, 3 = (x, y) coords of spheres, i.e. corners of rhombus
### Column 4, 5 = (x, y) where (x-0, y-0) defines unit vectors 

def findall_corners(): #given 2 of the 4 corners of each rhombus, find the other 2
	allcorners_x = [] 
	allcorners_y = []
	for i in range(0,len(corners_y)): 
		if corners_y[i]-coords_concat[i,1] == 0:
		 	m = 1e30
		else:
			m = -((corners_y[i]-coords_concat[i,1])/(corners_x[i]-coords_concat[i,0]))**-1
		b = coords_concat[i,1]-m*coords_concat[i,0]
		refl_x = ((1-m**2)*corners_x[i]+2*m*corners_y[i]-2*m*b)/(m**2+1)
		refl_y = ((m**2-1)*corners_y[i]+2*m*corners_x[i]+2*b)/(m**2+1)
		corner_this_iter_x = np.array([corners_x[i], refl_x])
		corner_this_iter_y = np.array([corners_y[i], refl_y])
		allcorners_x = np.append(corner_this_iter_x, allcorners_x)
		allcorners_y = np.append(corner_this_iter_y, allcorners_y)
	allcorners = np.column_stack((allcorners_x, allcorners_y))

	### sort corners
	corners_sorted = np.zeros((len(allcorners), 2))
	count = 0
	for rhomb_i in range(0, numRhombs):
		for corner_i in range(0,len(allcorners)):
			if np.round(np.linalg.norm(allcorners[corner_i,:] - coords_concat[rhomb_i,:]), 5) == np.round(117.55705, 5):
				corners_sorted[count,:] = allcorners[corner_i,:]
				count = count+1
			if np.round(np.linalg.norm(allcorners[corner_i,:] - coords_concat[rhomb_i,:]), 5) == np.round(161.80339887, 5):
				corners_sorted[count,:] = allcorners[corner_i,:]
				count = count+1
	return corners_sorted


def findeach_theta(): #find the rotation, theta of each rhombus, angle between long axis dipole, and vertical
	thetas = []
	allcorners = findall_corners()
	for rhomb_i in range(0, numRhombs):
		for corner_i in range(0, len(allcorners)):
			if np.round(np.linalg.norm(allcorners[corner_i,:] - coords_concat[rhomb_i,:]), 5) == np.round(161.80339887, 5):
				theta = np.pi - np.arctan2(allcorners[corner_i,0] - coords_concat[rhomb_i,0], allcorners[corner_i,1] - coords_concat[rhomb_i,1])
				thetas = np.append(thetas, theta)
				break
	return thetas

def rotate_indmodes():
	thetas = findeach_theta()
	allcorners = findall_corners()
	sphere_centers = allcorners

	Dl_vec_dirs = np.zeros((len(allcorners), 2))
	Ds_vec_dirs = np.zeros((len(allcorners), 2))
	Q_vec_dirs = np.zeros((len(allcorners), 2))

	dip_longaxis = np.loadtxt('parameter_files/rhomb_mode_DL.txt',skiprows=1)[:,2:4]
	dip_shortaxis = np.loadtxt('parameter_files/rhomb_mode_DS.txt',skiprows=1)[:,2:4]
	quad = np.loadtxt('parameter_files/rhomb_mode_Q.txt',skiprows=1)[:,2:4]

	for rhomb_i in range(0, numRhombs):

		theta = thetas[rhomb_i]
		Dl_vec_dirs[rhomb_i*4 : rhomb_i*4+4, 0] = dip_longaxis[:,0]*np.cos(theta) - dip_longaxis[:,1]*np.sin(theta)
		Dl_vec_dirs[rhomb_i*4 : rhomb_i*4+4, 1] = dip_longaxis[:,0]*np.sin(theta) + dip_longaxis[:,1]*np.cos(theta)
		
		Ds_vec_dirs[rhomb_i*4 : rhomb_i*4+4, 0] = dip_shortaxis[:,0]*np.cos(theta) - dip_shortaxis[:,1]*np.sin(theta)
		Ds_vec_dirs[rhomb_i*4 : rhomb_i*4+4, 1] = dip_shortaxis[:,0]*np.sin(theta) + dip_shortaxis[:,1]*np.cos(theta)
		
		Q_vec_dirs[rhomb_i*4 : rhomb_i*4+4, 0] = quad[:,0]*np.cos(theta) - quad[:,1]*np.sin(theta)
		Q_vec_dirs[rhomb_i*4 : rhomb_i*4+4, 1] = quad[:,0]*np.sin(theta) + quad[:,1]*np.cos(theta)
 
	return sphere_centers, Dl_vec_dirs, Ds_vec_dirs, Q_vec_dirs

sphere_centers, Dl_vec_dirs, Ds_vec_dirs, Q_vec_dirs = rotate_indmodes()
#print Dl_vec_dirs

def plot_indmodes(whichmode):
	xcoord = sphere_centers[:,0]; ycoord = sphere_centers[:,1]; 	
	if whichmode == 'D_l':
		xvec = Dl_vec_dirs[:,0]; yvec = Dl_vec_dirs[:,1]
	if whichmode == 'D_s':
		xvec = Ds_vec_dirs[:,0]; yvec = Ds_vec_dirs[:,1]
	if whichmode == 'Q':
		xvec = Q_vec_dirs[:,0]; yvec = Q_vec_dirs[:,1]
	plt.scatter(xcoord, ycoord)
	plt.quiver(xcoord, ycoord, xvec, yvec); 
	plt.axis('equal')
	plt.show()

#plot_indmodes(whichmode='D_l')

def final_writing():
	sphere_centers, Dl_vec_dirs, Ds_vec_dirs, Q_vec_dirs = rotate_indmodes()
	allcoords = np.zeros((len(sphere_centers),2))
	for i in range(0, numRhombs):
		allcoords[i*4 : i*(4)+4,:] = coordinates[i,:]
	final_write = np.column_stack((allcoords, sphere_centers, Dl_vec_dirs, Ds_vec_dirs, Q_vec_dirs))
	print final_write.shape
	### Writing time ###
	file = open(str('inputs_')+str(filename)+str('_0211.txt'),'w')
	file.write( 'Rhomb Center [nm]' + '\t' + '\t' + 'Dip Center [nm]' +  '\t' +  '\t' +  '\t' +'D_l [nm]' + '\t' +'D_s [nm]' + '\t' +'Q [nm]' + '\t' + '\n')
	for j in range(0, len(final_write)):
		file.write("%.3f" % final_write[j,:][0] + '\t' + "%.3f" % final_write[j,:][1] + '\t' +
				   "%.3f" % final_write[j,:][2] + '\t' + "%.3f" % final_write[j,:][3] + '\t' + 
				   "%.5f" % final_write[j,:][4] + '\t' + "%.5f" % final_write[j,:][5] + '\t' + 
				   "%.5f" % final_write[j,:][6] + '\t' + "%.5f" % final_write[j,:][7] + '\t' + 
				   "%.5f" % final_write[j,:][8] + '\t' + "%.5f" % final_write[j,:][9] + '\n')
	file.close()


final_writing()



