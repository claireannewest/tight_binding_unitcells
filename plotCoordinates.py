import numpy as np
import matplotlib.pyplot as plt
import yaml
open_param_file = open('parameter_files/parameters.yaml')
param = yaml.load(open_param_file)
numparticles = 1

coordinates = np.loadtxt('parameter_files/coords_trimerconcave.txt')
base_vectors = np.loadtxt('parameter_files/basevecs_trimerconcave.txt')

allcorners_x = []
allcorners_y = []
corners_this_iter_x = []
corners_this_iter_y =[]
allsides_x = []
allsides_y = []
longaxis_x = []
longaxis_y = []
shortaxis_x = []
shortaxis_y = []
coordinates = np.concatenate((coordinates, coordinates),axis=0)

# y1 = coordinates[1] + base_vectors[1]
# x1 = coordinates[0] + base_vectors[0]

corners_x = base_vectors[:,0]+coordinates[:,0]

corners_y = base_vectors[:,1]+coordinates[:,1]

### find all corners ### 
for i in range(0,len(corners_y)): 
	if corners_y[i]-coordinates[i,1] == 0:
	 	m = 1e30
	else:
		m = -((corners_y[i]-coordinates[i,1])/(corners_x[i]-coordinates[i,0]))**-1
	b = coordinates[i,1]-m*coordinates[i,0]
	refl_x = ((1-m**2)*corners_x[i]+2*m*corners_y[i]-2*m*b)/(m**2+1)
	refl_y = ((m**2-1)*corners_y[i]+2*m*corners_x[i]+2*b)/(m**2+1)
	corner_this_iter_x = np.array([corners_x[i], refl_x])
	corner_this_iter_y = np.array([corners_y[i], refl_y])
	allcorners_x = np.append(corner_this_iter_x, allcorners_x)
	allcorners_y = np.append(corner_this_iter_y, allcorners_y)
allcorners = np.column_stack((allcorners_x, allcorners_y))

# ### find all midpoints ### 
# for i in range(0,len(allcorners_x)): 
# 	for j in range(0,len(allcorners_x)): 
# 		current_corner = np.array([allcorners_x[i], allcorners_y[i]])
# 		next_corner = np.array([allcorners_x[j], allcorners_y[j]])
# 		if np.abs(np.round(np.linalg.norm(current_corner - next_corner))) == 200:
# 			side_x = (current_corner[0]+next_corner[0])/2
# 			side_y = (current_corner[1]+next_corner[1])/2

# 			allsides_x = np.append(side_x, allsides_x) 
# 			allsides_y = np.append(side_y, allsides_y) 
# allsides = np.unique(np.column_stack((allsides_x, allsides_y)),axis=0)

# ### find all dipole directions ###

# for i in range(0,len(allsides)):
# 	for j in range(0,len(allsides)):
# 		difference = np.round(np.linalg.norm(allsides[i,:]-allcorners[j,:]))
# 		if difference == 100:
# 			longaxis = allcorners[j,:] - allsides[i,:]
		# 	longaxis_x = np.append(longaxis[0], longaxis_x)
		# 	longaxis_y = np.append(longaxis[1], longaxis_y)
		# 	if allcorners[j,0]-allsides[i,0] == 0:
		# 		m_shortaxis = 0
		# 	else:
		# 		m_shortaxis = -((allcorners[j,1]-allsides[i,1])/(allcorners[j,0]-allsides[i,0]))**(-1)

# 			b_shortaxis = allsides[i,1]-m_shortaxis*allsides[i,0]
# 			print b_shortaxis
# 			shortaxis_x = np.append(coordinates[0]-allsides[i,0], shortaxis_x)
# 			shortaxis_y = np.append((m_shortaxis*coordinates[0] + b_shortaxis)-allsides[i,1], shortaxis_y)
# 			break

# longaxis = np.column_stack((longaxis_x, longaxis_y))
# shortaxis = np.column_stack((shortaxis_x, shortaxis_y))


# fig = plt.figure(1, figsize=[3.,3.])
# ax = plt.gca()
# plt.xlim([-400,400])
# plt.ylim([-400,400])
# ax.set_aspect('equal', adjustable='box')

for i in range(0, 1):#len(allsides)):
	# plt.arrow(allsides[i,0],allsides[i,1], longaxis[i,0], 
	# 	longaxis[i,1],head_width=10,color='k')

	# plt.arrow(allsides[i,0],allsides[i,1], shortaxis[i,0],
	# 	shortaxis[i,1],head_width=10,color='purple')

	print('coords', coordinates[:,0].shape)
	print('corners', allcorners_x)
	print('corners', allcorners_y)

	plt.scatter(coordinates[:,0], coordinates[:,1],color='k')
	plt.scatter(allcorners[:,0], allcorners[:,1], color='r')
#	plt.scatter(allsides[i,0], allsides[i,1], color='b')

	plt.xlim([-500,500])
	plt.ylim([-500,500])
	plt.show()

# # file = open('inputs_prolates.txt','w')
# # rhomb_centers = np.vstack((coordinates,coordinates,coordinates,coordinates,coordinates,coordinates,coordinates,coordinates))
# # allsides = np.vstack((allsides, allsides))
# # all_dip_directions = np.vstack((longaxis, shortaxis))
# # file.write( 'Rhomb Center [nm]' + '\t' + 'Dip Center [nm]' + '\t' + '\t' + '\t' + 'Dipole directions [nm]' + '\n')
# # for j in range(0, len(rhomb_centers)):
# # 	file.write("%.5f" % rhomb_centers[j,:][0] + '\t' + "%.5f" % rhomb_centers[j,:][1] + '\t' +
# # 			   "%.5f" % allsides[j,:][0] + '\t' + "%.5f" % allsides[j,:][1] + '\t' + 
# # 				"%.5f" % all_dip_directions[j,:][0] + '\t' + "%.5f" % all_dip_directions[j,:][1] + '\n')
# # file.close()

# file = open('inputs_spheres.txt','w')
# rhomb_centers = np.vstack((coordinates,coordinates,coordinates,coordinates,coordinates,coordinates,coordinates,coordinates))
# allcorners = np.vstack((allcorners, allcorners))
# all_dip_directions = np.vstack((longaxis, shortaxis))
# file.write( 'Rhomb Center [nm]' + '\t' + 'Dip Center [nm]' + '\t' + '\t' + '\t' + 'Dipole directions [nm]' + '\n')
# for j in range(0, len(rhomb_centers)):
# 	file.write("%.5f" % rhomb_centers[j,:][0] + '\t' + "%.5f" % rhomb_centers[j,:][1] + '\t' +
# 			   "%.5f" % allcorners[j,:][0] + '\t' + "%.5f" % allcorners[j,:][1] + '\t' + 
# 				"%.5f" % all_dip_directions[j,:][0] + '\t' + "%.5f" % all_dip_directions[j,:][1] + '\n')
# file.close()



