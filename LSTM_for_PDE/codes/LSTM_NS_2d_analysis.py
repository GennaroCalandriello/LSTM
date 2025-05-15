# LSTM_NS_2d_analysis.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from temp2 import LSTM_PINN_NS2D, sphere_noslip_mask
import hyperpar as hp

# 1) Hyperparameters & device
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = hp.HIDDEN_SIZE
num_layers  = hp.NUM_LAYERS
x_lb, x_ub  = hp.X_LB, hp.X_UB
y_lb, y_ub  = hp.Y_LB, hp.Y_UB
t_lb, t_ub  = hp.T_LB, 200

Nx, Ny, Nt  = 200, 200, 200      # grid resolution
batch_size  = 200                # evaluation batch size
xc, yc, r   = 0.5, 0.0, 0.4      # must match temp2.py

# 2) Load trained model
model = LSTM_PINN_NS2D(hidden_size=hidden_size,
                       num_layers=num_layers).to(device)
model.load_state_dict(torch.load(
    "models/LSTM_NS2d_improved.pth", map_location=device))
model.eval()

# 3) Build (x,y,t) grid
x = np.linspace(x_lb, x_ub, Nx)
y = np.linspace(y_lb, y_ub, Ny)
t = np.linspace(t_lb, t_ub, Nt)
Xg, Yg, Tg = np.meshgrid(x, y, t, indexing='ij')  # shape [Nx,Ny,Nt]

# 4) Obstacle mask on XY plane (True=fluid region)
mask_xy = ~sphere_noslip_mask(
    torch.tensor(Xg[:,:,0], dtype=torch.float32),
    torch.tensor(Yg[:,:,0], dtype=torch.float32)
).cpu().numpy()

# 5) Flatten to feed the network
x_flat = torch.tensor(Xg.ravel(), dtype=torch.float32).view(-1,1).to(device)
y_flat = torch.tensor(Yg.ravel(), dtype=torch.float32).view(-1,1).to(device)
t_flat = torch.tensor(Tg.ravel(), dtype=torch.float32).view(-1,1).to(device)

# derive φ flatten
phi_flat = (torch.sqrt((x_flat-xc)**2 + (y_flat-yc)**2) - r)

# 6) Evaluate u over the grid in batches
u_list = []
with torch.no_grad():
    for i in range(0, x_flat.shape[0], batch_size):
        xb  = x_flat[i : i+batch_size]
        yb  = y_flat[i : i+batch_size]
        tb  = t_flat[i : i+batch_size]
        phb = phi_flat[i : i+batch_size]
        u_b, _, _ = model(xb, yb, tb, phb)
        u_list.append(u_b.cpu().numpy())

u_flat = np.vstack(u_list)           # (Nx*Ny*Nt, 1)
U_xyz   = u_flat.reshape(Nx, Ny, Nt) # [Nx,Ny,Nt]
U_txy   = np.transpose(U_xyz, (2,0,1))   # [Nt,Nx,Ny]
U_mask  = np.where(mask_xy[None,:,:], U_txy, np.nan)

# 7) Precompute vmin/vmax for stable colorbar
vmin, vmax = np.nanmin(U_mask), np.nanmax(U_mask)

# 8) Animate u(x,y) over time
fig, ax = plt.subplots(figsize=(6,5))
pcm = ax.pcolormesh(
    x, y, U_mask[0].T,
    shading='auto', cmap='viridis',
    vmin=vmin, vmax=vmax
)
cbar = fig.colorbar(pcm, ax=ax, label='u(x,y)')
ax.set_xlabel('x')
ax.set_ylabel('y')

def update(frame):
    pcm.set_array(U_mask[frame].T.ravel())
    ax.set_title(f't = {t[frame]:.3f}')
    return pcm,

ani = animation.FuncAnimation(
    fig, update, frames=Nt, interval=100, blit=True
)

# Save GIF
ani.save("u_evolution.gif", writer='pillow', fps=10)
plt.close(fig)

if __name__ == "__main__":
    print("✔ Saved u_evolution.gif")
