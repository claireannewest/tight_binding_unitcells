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
numIndModes = (inputs.shape[1]-4)/2 #take the num of columns in inputs, subtract the first four location columns, then divide by two because each mode has an x,y

def seeVectors(mode):
    w = np.real(final_eigvals[mode])
    v = np.real(final_eigvecs[:,mode])

    sph_ycoords = dip_centers[:,0]
    sph_zcoords = dip_centers[:,1]  
   
    plt.subplot(1,3*numRhombs,mode+1)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    plt.title('%.2f eV' % (w), fontsize=18)
    plt.scatter(sph_ycoords, sph_zcoords,c='blue',s=50)

    for rhomb_i in range(0, int(numRhombs)):
        DL_i = D_l_vecs[int(4*rhomb_i) : int(4*(rhomb_i+1)), :]
        DS_i = D_s_vecs[int(4*rhomb_i) : int(4*(rhomb_i+1)), :]
        Q_i  = Q_vecs[int(4*rhomb_i) : int(4*(rhomb_i+1)), :]
        mag_mode = (v[int(numIndModes*rhomb_i)]*DL_i + v[int(numIndModes*rhomb_i+1)]*DS_i +v[int(numIndModes*rhomb_i+2)]*Q_i)

        ymin = min(sph_ycoords)-1E-5; ymax = max(sph_ycoords)+1E-5
        zmin = min(sph_zcoords)-1E-5; zmax = max(sph_zcoords)+1E-5
        plt.quiver(sph_ycoords[4*rhomb_i : 4*(rhomb_i+1)], sph_zcoords[4*rhomb_i : 4*(rhomb_i+1)], mag_mode[:,0], mag_mode[:,1], pivot='mid', 
            width=.5, #shaft width in arrow units 
            scale=1., 
            headlength=5,
            headwidth=5.,#5.8
            minshaft=4., #4.1
            minlength=.1)
    plt.xlim([ymin, ymax])
    plt.ylim([zmin, zmax])
    plt.yticks([])
    plt.xticks([])
    #plt.show()
    return w, mag_mode

final_eigvals, final_eigvecs = interate()


fig = plt.figure(num=None, figsize=(12, 3), dpi=80, facecolor='w', edgecolor='k')   
for mode in range(0,int(numIndModes*numRhombs)):
    seeVectors(mode=mode)
plt.show()
    
def seeFields(mode):
    w = np.real(final_eigvals[mode])
    v = np.real(final_eigvecs[:,mode])

    sph_xcoords = 0*dip_centers[:,0]
    sph_ycoords = dip_centers[:,0]
    sph_zcoords = dip_centers[:,1]  
    sphere_origins = np.column_stack((sph_xcoords, sph_ycoords, sph_zcoords))
    p = np.zeros((int(numRhombs*4), 3))
    for rhomb_i in range(0, int(numRhombs)):
        DL_i = D_l_vecs[4*(rhomb_i) : 4*(rhomb_i+1), :]
        DS_i = D_s_vecs[4*(rhomb_i) : 4*(rhomb_i+1), :]
        Q_i  = Q_vecs[4*(rhomb_i)   : 4*(rhomb_i+1), :]
        p[int(rhomb_i*4) : int((rhomb_i+1)*4),1:3] = (v[int(numIndModes*rhomb_i)]*DL_i + v[int(numIndModes*rhomb_i+1)]*DS_i + v[int(numIndModes*rhomb_i+2)]*Q_i)

    ymin = min(sphere_origins[:,1])-2E-5; ymax = max(sphere_origins[:,1])+2E-5
    zmin = min(sphere_origins[:,2])-2E-5; zmax = max(sphere_origins[:,2])+2E-5

    x = 60e-07; 
    numPoints = 71
    y = np.linspace(ymin, ymax, numPoints ); z = np.linspace(zmin, zmax, numPoints )

    ### Efield for every dipole, [ which dipole, which y point, which z point ] ###
    Ex_field = np.zeros((int(4*numRhombs), int(numPoints), int(numPoints)),dtype=complex)
    Ey_field = np.zeros((int(4*numRhombs), int(numPoints), int(numPoints)),dtype=complex)
    Ez_field = np.zeros((int(4*numRhombs), int(numPoints), int(numPoints)),dtype=complex)
    

    for which_dipole in range(0, int(4*numRhombs)):
        for which_y in range(0, numPoints ):
            for which_z in range(0, numPoints ):
                xval = x
                yval = y[which_y]
                zval = z[which_z]
                k = w/hbar_eVs/c
                point = np.array([xval, yval, zval])
                r = point - sphere_origins
                nhat = r/np.linalg.norm(r)
                nhat_dot_p = np.sum(nhat*p,axis=1)[:,np.newaxis]
                magr = np.linalg.norm(r,axis=1)[:,np.newaxis]
                nearField = ( 3*nhat * nhat_dot_p - p ) / magr**3

        #         intermedField1 = 1j*k*(3*rhat1 - np.dot(rhat1,p1)) / (np.linalg.norm(r1))**2
        #         farField1 = k**2*(rhat1 - np.dot(rhat1,p1)) / (np.linalg.norm(r1))
                Ex_field[which_dipole, which_z, which_y] = nearField[which_dipole,0]#(nearField1 + intermedField1 + farField1 ) * np.exp(1j*k*np.linalg.norm(r1))
                Ey_field[which_dipole, which_z, which_y] = nearField[which_dipole,1]#(nearField1 + intermedField1 + farField1 ) * np.exp(1j*k*np.linalg.norm(r1))
                Ez_field[which_dipole, which_z, which_y] = nearField[which_dipole,2]#(nearField1 + intermedField1 + farField1 ) * np.exp(1j*k*np.linalg.norm(r1))

    whichsphere = 1

    Extot = np.real(Ex_field[whichsphere,:,:]+Ex_field[1,:,:]+Ex_field[2,:,:]+Ex_field[3,:,:]+Ex_field[4,:,:]+Ex_field[5,:,:]+Ex_field[6,:,:]+Ex_field[7,:,:])
    #Eytot = np.real(Ey_field[whichsphere,:,:])#+Ey_field[1,:,:]+Ey_field[2,:,:]+Ey_field[3,:,:]+Ey_field[4,:,:]+Ey_field[5,:,:]+Ey_field[6,:,:]+Ey_field[7,:,:])
    #Eztot = np.real(Ez_field[whichsphere,:,:])

    plt.imshow(Extot, 
        cmap='seismic',
        origin='lower',
        extent=[ymin,ymax,zmin,zmax]
        )

    plt.scatter(sphere_origins[:,1], sphere_origins[:,2],c='black',s=30)
    plt.quiver(sphere_origins[:,1], sphere_origins[:,2], p[:,1], p[:,2], pivot='mid', 
        width=0.1, #shaft width in arrow units 
        scale=2., 
        headlength=4,
        headwidth=5.8,
        minshaft=4.1, 
        minlength=.1)
    plt.quiver(sphere_origins[whichsphere,1], sphere_origins[whichsphere,2], p[whichsphere,1], p[whichsphere,2], color='green',pivot='mid', width=0.1,scale=2.,headlength=4,headwidth=5.8,minshaft=4.1, minlength=.1)
    plt.show()

#seeFields(mode=0)

#for mode in range(0,int(numIndModes*numRhombs)):
#    seeFields(mode=mode)
#plt.show()
