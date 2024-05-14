import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import calfem.utils as cfu
import calfem.core as cfc
import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')

E = 5                   #Young's modulus, E [GPa]
v = 0.36                #Poisson's ratio, ν [-]
alpha = 0.00006         #Expansion coefficient, α [1/K]
p = 540                 #Density, ρ [kg/m3]
cp = 3600               #Specific heat, cp [J/kg-K]
k = 80                  #Thermal conductivity, k [W/m-K]
t = 1.6                 #Thickness in meters 
D = k*np.eye(2)

a_c = 120 # W/m2K
T_cool = 277 #Kelvin
T_hot = 285 #Kelvin
T_inf = 293 #Kelvin

g = cfg.Geometry()

g.point([0.0, 0.0]) # point 0
g.point([0.0, 200.0]) # point 1
g.point([400.0, 200.0]) # point 
g.point([300.0, 0.0]) # point 3

g.point([0, 125]) # point 4
g.point([0, 100]) # center circle 1; point 5
g.point([25, 100])# point 6
g.point([0, 75]) # point 7


g.point([87.5, 125]) # point 8
g.point([87.5, 100]) # center circle 2; point 9
g.point([112.5, 100]) # point 10
g.point([87.5, 75]) # point 11
g.point([62.5, 100]) # point 12



g.point([175, 125]) # point 13
g.point([175, 100]) # center circle 3; point 14
g.point([200, 100]) #  point 15
g.point([175, 75]) # point 16
g.point([150, 100]) # point 17

g.point([262.5, 125]) # point 18
g.point([262.5, 100]) # center circle 3; point 19
g.point([287.5, 100]) #  point 20
g.point([262.5, 75]) # point 21
g.point([237.5, 100]) # point 22


nothing = 10
hot = 30
cool = 20
ambient = 40



g.spline([4, 1], marker = nothing) #top left 
g.spline([1, 2], marker  = ambient) #Top row
g.spline([2, 3], marker = nothing) #Right slope
g.spline([3, 0], marker = nothing) #Bottom row
g.spline([0, 7], marker = nothing) #Left bottom

#circle 1
g.circle([4, 5, 6], marker = hot) 
g.circle([6, 5, 7], marker = hot)

#circle 2
g.circle([8, 9, 10], marker = cool) 
g.circle([10, 9, 11], marker = cool)
g.circle([11, 9, 12], marker = cool) 
g.circle([12, 9, 8], marker = cool)

#circle 3
g.circle([13, 14, 15], marker = hot) 
g.circle([15, 14, 16], marker = hot)
g.circle([16, 14, 17], marker = hot) 
g.circle([17, 14, 13], marker = hot)

#circle 4
g.circle([18, 19, 20], marker = cool) 
g.circle([20, 19, 21], marker = cool)
g.circle([21, 19, 22], marker = cool) 
g.circle([22, 19, 18], marker = cool)


g.surface([0, 1, 2, 3, 4, 6, 5], ([7, 8,9,10], [11,12,13,14], [15,16,17,18]))



mesh = cfm.GmshMesh(g)

mesh.el_type = 2         # Element type is quadrangle
mesh.dofs_per_node = 1     # Degrees of freedom per node
mesh.el_size_factor = 50 # Element size Factor

coords, edof, dofs, bdofs, elementmarkers = mesh.create()


nDofs = np.size(dofs)
ex, ey = cfc.coordxtr(edof, coords, dofs)
bc = np.array([],'i')
bcVal = np.array([],'f')
ex, ey = cfc.coordxtr(edof, coords, dofs)

K = np.zeros((len(coords), len(coords)))
f = np.zeros([len(edof),1])
print(edof)
print(bdofs)

for element in range(len(edof)):
    ex_n = np.array(ex[element])
    ey_n = np.array(ey[element])

    Cmat = np.vstack((np.ones((3, )), ex_n, ey_n))
    A = np.linalg.det(Cmat)/2
    B = np.matrix([[ey_n[1] - ey_n[2], ey_n[2] - ey_n[0], ey_n[0] - ey_n[1]], 
                  [ex_n[2] - ex_n[1], ex_n[0] - ex_n[2], ex_n[1] - ex_n[0]]])
                  
    B_e = A*B
    K_e = k * np.matmul(B_e.transpose(1, 0), B_e) * t * A
    cfc.assem(edof, K, K_e)

    #Boundary conditions
    #Top
    for node in range(3):

        if (edof[element][node] and edof[element][(node+1)%3]) in bdofs[hot]: #If two points on the boundary are on the hot battery
            distance =np.linalg.norm(coords[element][node], coords[element][(node+1)%3])
            f[edof[element][node]] +=  T_hot*a_c*t*distance
            f[edof[element][(node+1)%3]] +=  T_hot*a_c*t*distance
        if (edof[element][node] and edof[element][(node+1)%3]) in bdofs[cool]: #If two points on the boundary are on the cool battery
            distance = np.linalg.norm(coords[element][node], coords[element][(node+1)%3])
            f[edof[element][node]] +=  T-cool*a_c*t*distance
            f[edof[element][(node+1)%3]] +=  T_hot*a_c*t*distance
        if (edof[element][node] and edof[element][(node+1)%3]) in bdofs[ambient]:
            distance = np.linalg.norm(coords[element][node], coords[element][(node+1)%3])
            f[edof[element][node]] +=  T-cool*a_c*t*distance
            f[edof[element][(node+1)%3]] +=  T_hot*a_c*t*distance
        

    









bc, bcVal = cfu.applybc(bdofs, bc, bcVal, elementmarkers, 0.0, 0)
bc, bcVal = cfu.applybc(bdofs, bc, bcVal, elementmarkers, 0.0, 0)

#cfu.applyforce(bdofs, f, 7, 10e5, 1)

a,r = cfc.solveq(K,f,bc,bcVal)


#BOUNDERY CONDITIONS


cfv.figure()

# Draw the mesh.
cfv.drawMesh(
    coords=coords,
    edof=edof,
    dofs_per_node=mesh.dofsPerNode,
    el_type=mesh.elType,
    filled=True,
    title="Example 01"
        )

cfv.show_and_wait()

