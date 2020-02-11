import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')   
def rhombus():
	lat_space = 1 #distance between to points is lat_space in nm
	data_coord = np.loadtxt('parameter_files/coords_oneRhombus.txt')/lat_space
	basevecs = np.loadtxt('parameter_files/basevecs_oneRhombus.txt')/lat_space
	longcorner = basevecs[0:len(data_coord),:] + data_coord	
	side = 200./lat_space
	a = side*np.sin(54*np.pi/180.)
	b = side*np.cos(54*np.pi/180.)
	thickness = 28/lat_space #units are in lattice spacing
	x_go = 2#thickness/2
	y_go = side
	z_go = y_go
	Xval = []
	Yval = []
	Zval = []
	JAval = []
	count = 0
	for particle in range(0, 1):#len(data_coord[:,0])):
		# opp = basevecs[particle,1]
		# adj = basevecs[particle,0]
		# theta = np.arctan(opp/adj)
		# if theta < 0:
		# 	theta = np.pi + theta
		theta = 0#60*np.pi/180.
		#theta = np.pi
		# # y_cent = data_coord[particle,0]
		# z_cent = data_coord[particle,1]
		y_cent = data_coord[0]
		z_cent = data_coord[1]
		#x_range = np.linspace(-x_go, x_go, x_go*2+1)
		y_range = np.linspace(-y_go, y_go, y_go*2+1)
		z_range = np.linspace(-z_go, z_go,z_go*2+1)

		for i in range(0, len(y_range)):
			for j in range(0, len(z_range)):
		##		for k in range(0, len(x_range)):
		#	 		x = x_range[k]
			 		y = y_range[i]
					z = z_range[j]

					if np.abs((y)/a) + np.abs((z)/b) <= 1:
						count = count + 1
						yrot = y*np.cos(theta) - z*np.sin(theta)
					 	zrot = y*np.sin(theta) + z*np.cos(theta)
						y_shift = np.round(yrot + y_cent)
						z_shift = np.round(zrot + z_cent)
		#			 	Xval = np.append(Xval, x)
					 	Yval = np.append(Yval, y_shift)
					 	#print y_shift - data_coord[0]
					 	Zval = np.append(Zval, z_shift)
						JAval = np.append(JAval, count)

	Yval_shift = np.round(Yval - data_coord[0])
	Zval_shift = np.round(Zval - data_coord[1])
	#theta = -np.pi/2
	#yrot = Yval_shift*np.cos(theta) - Zval_shift*np.sin(theta)
	#zrot = Yval_shift*np.sin(theta) + Zval_shift*np.cos(theta)
	#plt.scatter(Yval_shift, Zval_shift, color='purple')
	#plt.scatter(yrot, zrot)
	plt.axis('equal')
	# dx_on_d = 1.0#1/20./d
	# dy_on_d = 1.0#1/11.75/d
	# dz_on_d = 1.0#1/16.18/d
	file = open(str('newershape.dat'),'w')
	# # file.write(str(' Rhombus Shape, 60 deg. DS=2') + '\n')
	# # file.write('\t' + str(int(max(JAval))) + str(' = number of dipoles in target') + '\n')
	# # file.write(str(' 1.000000 0.000000 0.000000 = A_1 vector') + '\n')
	# # file.write(str(' 0.000000 1.000000 0.000000 = A_2 vector') + '\n')
	# # file.write(str(' ') + str(dx_on_d) + str(' ') + str(dy_on_d) + str(' ')  + str(dz_on_d) + str(' ') + str('= (d_x,d_y,d_z)/d') + '\n')
	# # file.write(str(' 0.000000 0.000000 0.000000 = (x,y,z)/d') + '\n')
	# # file.write(str(' JA  IX  IY  IZ ICOMP(x,y,z)') + '\n')
	plt.scatter(Yval_shift, Zval_shift)
	#print Yval_shift.shape
	#for j in range(0, len(Yval)):
	#	file.write('\t' + str(int(JAval[j])) + '\t' + str(int(Yval_shift[j])) + '\t' + str(int(Zval_shift[j])) + '\t' + str(int(1)) + '\t' + str(int(1)) + '\t' + str(int(1)) + '\n')
		# file.write('\t' + str(int(JAval[j])) + '\t' + str(int(Xval[j])) + '\t' + str(int(Yval_shift[j])) + '\t' + str(int(Zval_shift[j])) + '\t' + str(int(1)) + '\t' + str(int(1)) + '\t' + str(int(1)) + '\n')
	#file.close()	

	#for i in range(0,len(data_coord[:,0])):
	#	plt.scatter(data_coord[i,0], data_coord[i,1], color='black')
	#	plt.scatter(data_coord[i,0]+basevecs[i,0], data_coord[i,1]+basevecs[i,1], color='blue')
rhombus()


#plt.xlim([-400, 400])
#plt.ylim([-400, 400])
plt.show()
