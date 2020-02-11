import numpy as np
import matplotlib.pyplot as plt

# dipole_origin = np.array([0,0,0])
# dipole_orientation = np.array([0,0,1])


# emode = np.loadtxt('output/rhomb_mode1.txt', skiprows=1)
# dipole = 1
# dipole_origin = np.array([0, emode[dipole,0], emode[dipole,1]])
# dipole_orientation = np.array([0, emode[dipole,2], emode[dipole,3]])
fig = plt.figure(1, figsize=[4,4])

def toy_dipole():
	just_y1 = []; just_z1 = []; just_E1 = []
	just_y2 = []; just_z2 = []; just_E2 = []
	just_y3 = []; just_z3 = []; just_E3 = []
	just_y4 = []; just_z4 = []; just_E4 = []

	origin1 = np.array([0,-1.2, 1.9])
	origin2 = np.array([0, 1.2, 1.9])
	origin3 = np.array([0, 0.0, 3.5])
	origin4 = np.array([0, 0.0, 0.3])

	### Mode 0 ###
	p1 = np.array([0,0,0.33])
	p2 = np.array([0,0,-0.33])
	p3 = np.array([0,0.63,0])
	p4 = np.array([0,-0.62,0])

	### Mode 1 ###
	p1 = np.array([0,0.03,0.22])*6
	p2 = np.array([0,-0.03,0.22])*6
	p3 = np.array([0,0,.73])*6
	p4 = np.array([0,0,.61])*6

	### Mode 2 ###
	# p1 = np.array([0, -.30, 0.02])#*6
	# p2 = np.array([0, .30, 0.02])#*6
	# p3 = np.array([0, 0, -.57])#*6
	# p4 = np.array([0, 0, .70])#*6

	# ### Mode 3 ###
	# p1 = np.array([0, 0.02, -0.01])#*6
	# p2 = np.array([0, 0.02, 0.01])#*6
	# p3 = np.array([0, 0.7, 0])#*6
	# p4 = np.array([0, 0.71, 0])#*6

	# ### Mode 4 ###
	# p1 = np.array([0, 0.7, -0.01])*6
	# p2 = np.array([0, 0.7, 0.01])*6
	# p3 = np.array([0, -0.03, 0])*6
	# p4 = np.array([0, -0.03, 0])*6

	# ### Mode 5 ###
	# p1 = np.array([0, 0, -0.58])#*6
	# p2 = np.array([0, 0, 0.58])#*6
	# p3 = np.array([0, 0.40, 0])#*6
	# p4 = np.array([0, -0.40, 0])#*6

	# ### Mode 6 ###
	# p1 = np.array([0, 0, 0.65])#*6
	# p2 = np.array([0, 0, 0.65])#*6
	# p3 = np.array([0, 0, -0.28])#*6
	# p4 = np.array([0, 0, -0.28])#*6

	### Mode 7 ###
	# p1 = np.array([0, -0.60, 0])#*6
	# p2 = np.array([0, 0.60, 0])#*6
	# p3 = np.array([0, 0, 0.37])#*6
	# p4 = np.array([0, 0, -0.37])#*6

	xmin = -5; xmax = 5
	ymin = -1.2 - 4; ymax = 1.2 + 4
	zmin = 0.3 - 4; zmax = 3.5 + 4

	x1 = np.linspace(xmin, xmax, 51 ); y1 = np.linspace(ymin, ymax, 51 ); z1 = np.linspace(zmin, zmax, 51 )
	x2 = np.linspace(xmin, xmax, 51 ); y2 = np.linspace(ymin, ymax, 51 ); z2 = np.linspace(zmin, zmax, 51 )
	x3 = np.linspace(xmin, xmax, 51 ); y3 = np.linspace(ymin, ymax, 51 ); z3 = np.linspace(zmin, zmax, 51 )
	x4 = np.linspace(xmin, xmax, 51 ); y4 = np.linspace(ymin, ymax, 51 ); z4 = np.linspace(zmin, zmax, 51 )

	x1_grid, y1_grid, z1_grid = np.meshgrid(x1, y1, z1)
	x2_grid, y2_grid, z2_grid = np.meshgrid(x2, y2, z2)
	x3_grid, y3_grid, z3_grid = np.meshgrid(x3, y3, z3)
	x4_grid, y4_grid, z4_grid = np.meshgrid(x4, y4, z4)

	all_points1 = np.column_stack((np.ravel(x1_grid), np.ravel(y1_grid), np.ravel(z1_grid)))
	all_points2 = np.column_stack((np.ravel(x2_grid), np.ravel(y2_grid), np.ravel(z2_grid)))
	all_points3 = np.column_stack((np.ravel(x3_grid), np.ravel(y3_grid), np.ravel(z3_grid)))
	all_points4 = np.column_stack((np.ravel(x4_grid), np.ravel(y4_grid), np.ravel(z4_grid)))

	E1_field = np.zeros((len(all_points1[:,0]), 3))
	E2_field = np.zeros((len(all_points2[:,0]), 3))
	E3_field = np.zeros((len(all_points3[:,0]), 3))
	E4_field = np.zeros((len(all_points4[:,0]), 3))

	for coord in range(0, len(all_points1[:,0])):
		p1_scale = p1
		r1 = all_points1[coord] - origin1
		if (np.linalg.norm(r1)) == 0:
			E1_field[coord, :] = 0
		else:
			rhat1 = r1/np.linalg.norm(r1)
			nearField = ( 3*rhat1*( np.dot(rhat1,p1_scale) ) - p1_scale ) / (np.linalg.norm(r1))**3
			E1_field[coord, :] = nearField

	for coord in range(0, len(all_points2[:,0])):
		p2_scale = p2
		r2 = all_points2[coord] - origin2
		if (np.linalg.norm(r2)) == 0:
			E2_field[coord, :] = 0
		else:
			rhat2 = r2/np.linalg.norm(r2)
			nearField = ( 3*rhat2*( np.dot(rhat2,p2_scale) ) - p2_scale ) / (np.linalg.norm(r2))**3
			E2_field[coord, :] = nearField
	
	for coord in range(0, len(all_points3[:,0])):
		p3_scale = p3
		r3 = all_points3[coord] - origin3
		if (np.linalg.norm(r3)) == 0:
			E3_field[coord, :] = 0
		else:
			rhat3 = r3/np.linalg.norm(r3)
			nearField = ( 3*rhat3*( np.dot(rhat3,p3_scale) ) - p3_scale ) / (np.linalg.norm(r3))**3
			E3_field[coord, :] = nearField

	for coord in range(0, len(all_points4[:,0])):
		p4_scale = p4
		r4 = all_points4[coord] - origin4
		if (np.linalg.norm(r4)) == 0:
			E4_field[coord, :] = 0
		else:
			rhat4 = r4/np.linalg.norm(r4)
			nearField = ( 3*rhat4*( np.dot(rhat4,p4_scale) ) - p4_scale ) / (np.linalg.norm(r4))**3
			E4_field[coord, :] = nearField

	#############################################

	for coord in range(0, len(all_points1[:,0])):
		if all_points1[coord,0] == 1:
			just_y1 = np.append(all_points1[coord,1], just_y1)
			just_z1 = np.append(all_points1[coord,2], just_z1)
			just_E1 = np.append(E1_field[coord,0], just_E1)

	for coord in range(0, len(all_points2[:,0])):
		if all_points2[coord,0] == 1:
			just_y2 = np.append(all_points2[coord,1], just_y2)
			just_z2 = np.append(all_points2[coord,2], just_z2)
			just_E2 = np.append(E2_field[coord,0], just_E2)

	for coord in range(0, len(all_points3[:,0])):
		if all_points3[coord,0] == 1:
			just_y3 = np.append(all_points3[coord,1], just_y3)
			just_z3 = np.append(all_points3[coord,2], just_z3)
			just_E3 = np.append(E3_field[coord,0], just_E3)

	for coord in range(0, len(all_points4[:,0])):
		if all_points4[coord,0] == 1:
			just_y4 = np.append(all_points3[coord,1], just_y4)
			just_z4 = np.append(all_points3[coord,2], just_z4)
			just_E4 = np.append(E4_field[coord,0], just_E4)

	Etot = (just_E1+just_E2+just_E3+just_E4).reshape((51,51),order='F')
	

	plt.imshow(Etot, cmap='seismic',
		vmin = -1,
		vmax = 1
		)
	plt.colorbar()
	print 'complete'
	# plt.xlim([ymin, ymax])
	# plt.ylim([zmin,ymax])
	# plt.axis('equal')
	plt.show()





toy_dipole()
fig.savefig('analytic_d.png')



# def toy_dipole():
# 	origin = np.array([0,0,0])
# 	p = np.array([0,0,1])
# 	x = np.linspace(-5, 5, 51 )
# 	y = np.linspace(-5, 5, 51 )
# 	z = np.linspace(-5, 5, 51 )
# 	x_grid, y_grid, z_grid = np.meshgrid(x, y, z)
# 	all_points = np.column_stack((np.ravel(x_grid), np.ravel(y_grid), np.ravel(z_grid)))
# 	E_field = np.zeros((len(all_points[:,0]), 3))
# 	for coord in range(0, len(all_points[:,0])):
# 		p_scale = p
# 		r = all_points[coord] - p_scale
# 		if (np.linalg.norm(r)) == 0:
# 			E_field[coord, :] = nearField
# 		else:
# 			rhat = r/np.linalg.norm(r)
# 			nearField = ( 3*rhat*( np.dot(rhat,p_scale) ) - p_scale ) / (np.linalg.norm(r))**3
# 			E_field[coord, :] = nearField

# 	just_y = []; just_z = []; just_E = []

# 	for coord in range(0, len(all_points[:,0])):
# 		if all_points[coord,0] == 1:
# 			just_y = np.append(all_points[coord,1], just_y)
# 			just_z = np.append(all_points[coord,2], just_z)
# 			just_E = np.append(E_field[coord,0], just_E)

# 	plt.scatter(just_y, just_z, c=just_E, s=10, cmap='seismic')
# 	plt.show()

# 	return just_y, just_z, just_E



def e_dipole(mode_bounds, origin,p):
	origin = origin
	x = np.linspace(origin[0]-2.0000e-5, origin[0]+2.0000e-5, 50 )
	y = np.linspace( min(mode_bounds[:,0])-1e-5, max(mode_bounds[:,0])+1e-5, 50 )
	z = np.linspace( min(mode_bounds[:,1])-1e-5, max(mode_bounds[:,1])+1e-5, 50 )
	x_grid, y_grid, z_grid = np.meshgrid(x, y, z)
	all_points = np.column_stack((np.ravel(x_grid), np.ravel(y_grid), np.ravel(z_grid)))
	E_field = np.zeros((len(all_points[:,0]), 3))
	for coord in range(0, len(all_points[:,0])):
		p_scale = p*5E-4
		r = all_points[coord] - origin
		if (np.linalg.norm(r)) == 0:
			E_field[coord, :] = nearField
		else:
			rhat = r/np.linalg.norm(r)
			print rhat
			nearField = ( 3*rhat*( np.dot(rhat,p_scale) ) - p_scale ) / (np.linalg.norm(r))**3
			E_field[coord, :] = nearField

	just_y = []; just_z = []; just_E = []

	for coord in range(0, len(all_points[:,0])):
		if all_points[coord,0] == 4.081632653061209e-07:
			just_y = np.append(all_points[coord,1], just_y)
			just_z = np.append(all_points[coord,2], just_z)
			just_E = np.append(E_field[coord,0], just_E)
	plt.scatter(just_y, just_z, c=just_E, s=10, cmap='seismic')
	print 'complete'
	plt.show()
	return just_y, just_z, just_E

# e_dipole(mode_bounds = emode, origin=dipole_origin,p=dipole_orientation)




def e_fields_plot():
	emode = np.loadtxt('output/rhomb_mode7.txt', skiprows=1)
	just_y, just_z, just_E = e_dipole(mode_bounds = emode, origin=np.array([0, emode[0,0], emode[0,1]]), p=np.array([0, emode[0,2], emode[0,3]]))
	E_tot = np.zeros(just_E.shape)
	p_all = np.array([emode[:,2], emode[:,3]])
	dipole_scaling = max(np.sqrt(emode[:,2]**2 + emode[:,3]**2))
	print p_all/dipole_scaling
	print 
	print p_all
	# for dipole in range(0,4):
	# 	dipole_origin = np.array([0, emode[dipole,0], emode[dipole,1]])
	# 	dipole_orientation = np.array([0, emode[dipole,2], emode[dipole,3]])
	# 	just_y, just_z, just_E = e_dipole(mode_bounds = emode, origin=dipole_origin, p=dipole_orientation)
	# 	E_tot = E_tot + just_E
	# 	plt.quiver(dipole_origin[1],dipole_origin[2],dipole_orientation[1],dipole_orientation[2], 
 #            pivot='mid', 
 #            width=0.1,
 #            scale=2.0, 
 #            headlength=4,
 #            headwidth=5.8,
 #            minshaft=4.1, 
 #            minlength=.1
 #            )
	# plt.scatter(just_y, just_z, c=E_tot, s=10, cmap='seismic')
	# plt.scatter(emode[0,0], emode[0,1],color='black')
	# plt.scatter(emode[1,0], emode[1,1],color='black')
	# plt.scatter(emode[2,0], emode[2,1],color='black')
	# plt.scatter(emode[3,0], emode[3,1],color='black')



	# print 'showing'
	# plt.xlim([min(emode[:,0])-1e-5, max(emode[:,0])+1e-5])
	# plt.ylim([min(emode[:,1])-1e-5, max(emode[:,1])+1e-5])
	# #plt.colorbar()
	# plt.show()

#e_fields_plot()
