import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1) Re-define the model architecture
class LSTMPINN(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=64, num_layers=2, out_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        h, _ = self.lstm(x)           # → (batch, seq_len, hidden_dim)
        y = self.fc(h)                # → (batch, seq_len, out_dim)
        return y

# 2) Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMPINN().to(device)
model.load_state_dict(torch.load("models/LSTM_PINN_NS2D.pth", map_location=device))
model.eval()

# 3) Build a spatial grid and time‐vector
nx, ny = 50, 50
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)            # shapes: (ny, nx)
xy_flat = np.stack([X.ravel(), Y.ravel()], axis=1)  # (nx*ny, 2)

nt = 100
T_final = 1.0
times = np.linspace(0, T_final, nt)

# 4) Utility to compute (U,V,ω) at a single time
def eval_uvomega(t_scalar):
    # prepare torch input
    t_col = np.full((nx*ny,1), t_scalar, dtype=np.float32)
    inp = np.concatenate([xy_flat.astype(np.float32), t_col], axis=1)
    with torch.no_grad():
        T = torch.from_numpy(inp).to(device).unsqueeze(1)  # (N,1,3)
        uvp = model(T).squeeze(1).cpu().numpy()           # (N,3)
    U = uvp[:,0].reshape((ny,nx))
    V = uvp[:,1].reshape((ny,nx))
    # finite‐difference vorticity
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dvdx = np.gradient(V, axis=1) / dx
    dudy = np.gradient(U, axis=0) / dy
    omega = dvdx - dudy
    return U, V, omega

# 5) Set up the figure
fig, ax = plt.subplots(figsize=(6,6))
U0, V0, ω0 = eval_uvomega(times[0])
# quiver for velocity
Q = ax.quiver(X, Y, U0, V0, pivot='middle', scale=50)
# colormap for vorticity
C = ax.imshow(ω0, origin='lower',
              extent=(0,1,0,1),
              cmap='RdBu', alpha=0.6)
cb = fig.colorbar(C, ax=ax, label='Vorticity')
title = ax.set_title(f"$t={times[0]:.2f}$")

def update(frame):
    t = times[frame]
    U, V, ω = eval_uvomega(t)
    Q.set_UVC(U, V)
    C.set_data(ω)
    title.set_text(f"$t={t:.2f}$")
    return Q, C, title

ani = FuncAnimation(fig, update, frames=nt, interval=50, blit=False)

# To save as MP4 or GIF, uncomment one of these:
# ani.save("ns2d_velocity_vorticity.mp4", writer="ffmpeg", dpi=150)
# ani.save("ns2d_velocity_vorticity.gif", writer="pillow", fps=20)

plt.show()
