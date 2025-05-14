# ns3d_mayavi.py

import numpy as np
from mayavi import mlab

# 1) Load precomputed data from your analysis step
#    Replace these with your actual loading code or pass them in.
#    U_masked: numpy array (Nt, Nz, Ny, Nx)
#    V_masked, W_masked same shape
#    x: length-Nx
#    y: length-Ny
#    z: length-Nz
data = np.load("ns3d_data.npz")  
U = data["U_masked"]
V = data["V_masked"]
W = data["W_masked"]
x = data["x"]
y = data["y"]
z = data["z"]
t = data["t"]  # length Nt

# 2) Choose a time index to visualize
t_idx = 10

# 3) Extract the 3D fields at that time
u = U[ t_idx ]  # shape (Nz,Ny,Nx)? or (Ny,Nx,Nz) depending on your ordering
v = V[ t_idx ]
w = W[ t_idx ]

# If your arrays are (Nt,Nz,Ny,Nx), you need to reorder:
#    u = U[t_idx].transpose(2,1,0)  # → (Nx,Ny,Nz)
#    v = V[t_idx].transpose(2,1,0)
#    w = W[t_idx].transpose(2,1,0)
# And build X,Y,Z via meshgrid(x,y,z,indexing='ij') → (Nx,Ny,Nz)

# Here we assume u,v,w are now (Nx,Ny,Nz):
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Replace NaNs inside obstacle by zero so VTK can ingest the field
u = np.nan_to_num(u)
v = np.nan_to_num(v)
w = np.nan_to_num(w)

# 4) Launch Mayavi figure
mlab.figure(size=(800,600), bgcolor=(1,1,1))

# 5) Create a vector field source
src = mlab.pipeline.vector_field(X, Y, Z, u, v, w)

# 6) Add an outline and axes for context
mlab.outline(color=(0,0,0))
mlab.axes(xlabel='x', ylabel='y', zlabel='z')

# 7) Add streamlines
#    Seed them on a plane at x = x_lb, spanning y,z
x0 = x[0]
plane = mlab.pipeline.scalar_cut_plane(src,
    plane_orientation='x_axes', slice_index=0)
plane.implicit_plane.widget.enabled = False

strm = mlab.pipeline.streamline(src,
    seedtype='plane',
    seed_visible=True,
    seed_scale=1.0,
    integration_direction='both',
    colormap='autumn')
strm.stream_tracer.maximum_propagation = 200
strm.tube_filter.radius = 0.005

# 8) (Optional) Add an isosurface of speed
speed = np.sqrt(u*u + v*v + w*w)
iso = mlab.pipeline.iso_surface(
    mlab.pipeline.scalar_field(X, Y, Z, speed),
    contours=[0.5*speed.max()],
    opacity=0.3,
    colormap='Blues'
)

# 9) Add a colorbar for speed if desired
mlab.scalarbar(object=iso, title='|u|')

# 10) Show the scene
mlab.view(azimuth=45, elevation=60, distance='auto')
mlab.show()
