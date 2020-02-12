import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure    
import yaml
from claires_tight_unitcells import interate

open_param_file = open('parameter_files/parameters.yaml')
param = yaml.load(open_param_file)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
eps_b = np.sqrt(param['n_b'])
c = param['constants']['c']
hbar_eVs = param['constants']['hbar_eVs']

inputs = np.loadtxt(param['inputs'],skiprows=1)
rhomb_centers = inputs[:,0:2]*1E-7
dip_centers = inputs[:,2:4]*1E-7
D_l_vecs = inputs[:,4:6]
D_s_vecs = inputs[:,6:8]
Q_vecs = inputs[:,8:10]
numRhombs = len(inputs)/4

total = interate()

def seeVectors(mode):
    w = np.real(total[:,0])
    v = np.real(total[:,1:])
    sph_ycoords = dip_centers[:,0]
    sph_zcoords = dip_centers[:,1]  
   
    plt.subplot(1,6,mode+1)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    evalue = np.sqrt(np.real(w[mode]))*hbar_eVs
    plt.title('%.2f eV' % (evalue), fontsize=18)
    plt.scatter(sph_ycoords, sph_zcoords,c='blue',s=50)

    for rhomb_i in range(0, numRhombs):
        DL_i = D_l_vecs[4*(rhomb_i) : 4*(rhomb_i)+4, :]
        DS_i = D_s_vecs[4*(rhomb_i) : 4*(rhomb_i)+4, :]
        Q_i  = Q_vecs[4*(rhomb_i) : 4*(rhomb_i)+4, :]
        mag_mode = (v[mode,3*rhomb_i]*DL_i + v[mode,3*rhomb_i+1]*DS_i +v[mode,3*rhomb_i+2]*Q_i)

        ymin = min(sph_ycoords)-1E-5; ymax = max(sph_ycoords)+1E-5
        zmin = min(sph_zcoords)-1E-5; zmax = max(sph_zcoords)+1E-5
        plt.quiver(sph_ycoords[4*rhomb_i : 4*rhomb_i+4], sph_zcoords[4*rhomb_i : 4*rhomb_i+4], mag_mode[:,0], mag_mode[:,1], pivot='mid', 
            width=0.1, #shaft width in arrow units 
            scale=3, 
            headlength=4,
            headwidth=5.,#5.8
            minshaft=2., #4.1
            minlength=.1)
    plt.xlim([ymin, ymax])
    plt.ylim([zmin, zmax])
    plt.yticks([])
    plt.xticks([])
    #plt.show()
    return w, mag_mode

#seeVectors(mode=0)

fig = plt.figure(num=None, figsize=(12, 3), dpi=80, facecolor='w', edgecolor='k')   
for mode in range(0,6):
    seeVectors(mode=mode)
    print mode
plt.show()

    
# def seeFields(mode):
#     sphere_origins = np.loadtxt(param['sphere_origins'],skiprows=2)*1E-7
#     w, mag_mode = seeVectors(mode=mode)
#     origin = np.transpose(np.array([np.zeros(8), sphere_origins[:,0], sphere_origins[:,1]]))
#     p = np.transpose(np.array([np.zeros(8), mag_mode[:,0], mag_mode[:,1]]))

#     ymin = min(sphere_origins[:,0])-2E-5; ymax = max(sphere_origins[:,0])+2E-5
#     zmin = min(sphere_origins[:,1])-2E-5; zmax = max(sphere_origins[:,1])+2E-5

#     x = 200e-07; y = np.linspace(ymin, ymax, 51 ); z = np.linspace(zmin, zmax, 51 )
#     x_grid, y_grid, z_grid = np.meshgrid(x, y, z)
#     all_points = np.column_stack((np.ravel(x_grid), np.ravel(y_grid), np.ravel(z_grid)))

#     E1_field = np.zeros((len(all_points[:,0]), 3),dtype=complex)
#     E2_field = np.zeros((len(all_points[:,0]), 3),dtype=complex)
#     E3_field = np.zeros((len(all_points[:,0]), 3),dtype=complex)
#     E4_field = np.zeros((len(all_points[:,0]), 3),dtype=complex)
#     E5_field = np.zeros((len(all_points[:,0]), 3),dtype=complex)
#     E6_field = np.zeros((len(all_points[:,0]), 3),dtype=complex)
#     E7_field = np.zeros((len(all_points[:,0]), 3),dtype=complex)
#     E8_field = np.zeros((len(all_points[:,0]), 3),dtype=complex)

#     for coord in range(0, len(all_points[:,0])):
#         omega = w[mode]/hbar_eVs
#         k = omega/c

#         r1 = all_points[coord] - origin[0,:]
#         p1 = p[0,:]

#         rhat1 = r1/np.linalg.norm(r1)
#         nearField1 = ( 3*rhat1*( np.dot(rhat1,p1) ) - p1 ) / (np.linalg.norm(r1))**3
#         intermedField1 = 1j*k*(3*rhat1 - np.dot(rhat1,p1)) / (np.linalg.norm(r1))**2
#         farField1 = k**2*(rhat1 - np.dot(rhat1,p1)) / (np.linalg.norm(r1))
#         E1_field[coord, :] = nearField1#(nearField1 + intermedField1 + farField1 ) * np.exp(1j*k*np.linalg.norm(r1))

#         r2 = all_points[coord] - origin[1,:]
#         p2 = p[1,:]
#         rhat2 = r2/np.linalg.norm(r2)
#         nearField2 = ( 3*rhat2*( np.dot(rhat2,p2) ) - p2 ) / (np.linalg.norm(r2))**3
#         intermedField2 = 1j*k*(3*rhat2 - np.dot(rhat2,p2)) / (np.linalg.norm(r2))**2
#         farField2 = k**2*(rhat2 - np.dot(rhat2,p2)) / (np.linalg.norm(r2))
#         E2_field[coord, :] = nearField2#(nearField2 + intermedField2 + farField2 ) * np.exp(1j*k*np.linalg.norm(r2))

#         r3 = all_points[coord] - origin[2,:]
#         p3 = p[2,:]
#         rhat3 = r3/np.linalg.norm(r3)
#         nearField3 = ( 3*rhat3*( np.dot(rhat3,p3) ) - p3 ) / (np.linalg.norm(r3))**3
#         intermedField3 = 1j*k*(3*rhat3 - np.dot(rhat3,p3)) / (np.linalg.norm(r3))**2
#         farField3 = k**2*(rhat3 - np.dot(rhat3,p3)) / (np.linalg.norm(r3))
#         E3_field[coord, :] = nearField3#(nearField3 + intermedField3 + farField3 ) * np.exp(1j*k*np.linalg.norm(r3))

#         r4 = all_points[coord] - origin[3,:]
#         p4 = p[3,:]
#         rhat4 = r4/np.linalg.norm(r4)
#         nearField4 = ( 3*rhat4*( np.dot(rhat4,p4) ) - p4 ) / (np.linalg.norm(r4))**3
#         intermedField4 = 1j*k*(3*rhat4 - np.dot(rhat4,p4)) / (np.linalg.norm(r4))**2
#         farField4 = k**2*(rhat4 - np.dot(rhat4,p4)) / (np.linalg.norm(r4))
#         E4_field[coord, :] = nearField4#(nearField4 + intermedField4 + farField4 ) * np.exp(1j*k*np.linalg.norm(r4))

#         r5 = all_points[coord] - origin[4,:]
#         p5 = p[4,:]
#         rhat5 = r5/np.linalg.norm(r5)
#         nearField5 = ( 3*rhat5*( np.dot(rhat5,p5) ) - p5 ) / (np.linalg.norm(r5))**3
#         intermedField5 = 1j*k*(3*rhat5 - np.dot(rhat5,p5)) / (np.linalg.norm(r5))**2
#         farField5 = k**2*(rhat5 - np.dot(rhat5,p5)) / (np.linalg.norm(r5))
#         E5_field[coord, :] = nearField5#(nearField5 + intermedField5 + farField5 ) * np.exp(1j*k*np.linalg.norm(r5))

#         r6 = all_points[coord] - origin[5,:]
#         p6 = p[5,:]
#         rhat6 = r6/np.linalg.norm(r6)
#         nearField6 = ( 3*rhat6*( np.dot(rhat6,p6) ) - p6 ) / (np.linalg.norm(r6))**3
#         intermedField6 = 1j*k*(3*rhat6 - np.dot(rhat6,p6)) / (np.linalg.norm(r6))**2
#         farField6 = k**2*(rhat6 - np.dot(rhat6,p6)) / (np.linalg.norm(r6))
#         E6_field[coord, :] = nearField6#(nearField6 + intermedField6 + farField6 ) * np.exp(1j*k*np.linalg.norm(r6))

#         r7 = all_points[coord] - origin[6,:]
#         p7 = p[6,:]
#         rhat7 = r7/np.linalg.norm(r7)
#         nearField7 = ( 3*rhat7*( np.dot(rhat7,p7) ) - p7 ) / (np.linalg.norm(r7))**3
#         intermedField7 = 1j*k*(3*rhat7 - np.dot(rhat7,p7)) / (np.linalg.norm(r7))**2
#         farField7 = k**2*(rhat7 - np.dot(rhat7,p7)) / (np.linalg.norm(r7))
#         E7_field[coord, :] = nearField7#(nearField7 + intermedField7 + farField7 ) * np.exp(1j*k*np.linalg.norm(r7))

#         r8 = all_points[coord] - origin[7,:]
#         p8 = p[7,:]
#         rhat8 = r8/np.linalg.norm(r8)
#         nearField8 = ( 3*rhat8*( np.dot(rhat8,p8) ) - p8 ) / (np.linalg.norm(r8))**3
#         intermedField8 = 1j*k*(3*rhat8 - np.dot(rhat8,p8)) / (np.linalg.norm(r8))**2
#         farField8 = k**2*(rhat8 - np.dot(rhat8,p8)) / (np.linalg.norm(r8))
#         E8_field[coord, :] = nearField8#(nearField8 + intermedField8 + farField8 ) * np.exp(1j*k*np.linalg.norm(r8))

#     Etot = E1_field[:,0] + E2_field[:,0] + E3_field[:,0] + E4_field[:,0] + E5_field[:,0] + E6_field[:,0] + E7_field[:,0] + E8_field[:,0]
#     plt.subplot(1,6,mode+1)
#     evalue = w[mode]
#     plt.title('%.2f eV' % (evalue), fontsize=14)
#     plt.scatter(all_points[:,1], all_points[:,2], c=np.real(Etot), 
#         s=10, cmap='seismic',
#         alpha=0.75,
#         vmin = min(np.real(Etot)),
#         vmax = max(np.real(Etot))
#         )

#     plt.scatter(sphere_origins[:,0], sphere_origins[:,1],c='black',s=10)
#     plt.quiver(sphere_origins[:,0], sphere_origins[:,1], mag_mode[:,0], mag_mode[:,1], pivot='mid', 
#         width=0.1, #shaft width in arrow units 
#         scale=1.55, 
#         headlength=4,
#         headwidth=5.8,
#         minshaft=4.1, 
#         minlength=.1)
#     plt.xlim([ymin, ymax])
#     plt.ylim([zmin, zmax])
#     plt.yticks([])
#     plt.xticks([])
#     #plt.axis('off')



# # for mode in range(0,6):
# #     seeFields(mode=mode)
# #     print mode
# # plt.show()

