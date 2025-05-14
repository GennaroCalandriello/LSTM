import os
import torch
import torch.nn as nn
import numpy as np
import hyperpar as hp

# 0) Disable CuDNN for double‐backward through LSTM
torch.backends.cudnn.enabled = False

# 1) Hyperparameters & device
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nu           = hp.NU
hidden_size  = hp.HIDDEN_SIZE
num_layers   = hp.NUM_LAYERS
lr           = hp.LR
epochs       = hp.EPOCHS

# Numbers of sample points
N_ic         = hp.N_IC
N_colloc     = hp.N_COLLOCATION
N_walls      = hp.N_WALLS
N_obs        = hp.N_OBSTACLE
N_inlet      = hp.N_INLET
N_outlet     = hp.N_OUTLET

# Loss weights (stronger obstacle enforcement)
lambda_ic     = hp.LAMBDA_IC
lambda_pde    = hp.LAMBDA_PDE
lambda_walls  = hp.LAMBDA_BC
lambda_obs    = hp.LAMBDA_OBS * 10
lambda_inlet  = hp.LAMBDA_INLET
lambda_outlet = hp.LAMBDA_OUTLET

# Domain bounds
x_lb, x_ub = hp.X_LB, hp.X_UB
y_lb, y_ub = hp.Y_LB, hp.Y_UB
t_lb, t_ub = hp.T_LB, 20

# Obstacle geometry
xc, yc, r = 0.5, 0.0, 0.4

def sphere_noslip_mask(x, y):
    """Mask of points INSIDE the cylinder."""
    return ((x - xc)**2 + (y - yc)**2) < r**2

def compute_phi(x, y):
    """Signed‐distance from obstacle boundary."""
    return torch.sqrt((x - xc)**2 + (y - yc)**2) - r

# ── Sampling utilities ──────────────────────────────────────────────────────────

def sample_outside(N):
    """Exactly N points uniformly outside the obstacle."""
    xs, ys, ts = [], [], []
    count = 0
    while count < N:
        m = N - count
        X = torch.rand(m,1, device=device)*(x_ub - x_lb) + x_lb
        Y = torch.rand(m,1, device=device)*(y_ub - y_lb) + y_lb
        T = torch.rand(m,1, device=device)*(t_ub - t_lb) + t_lb
        mask = ~sphere_noslip_mask(X, Y)
        if mask.any():
            Xm = X[mask].view(-1,1)
            Ym = Y[mask].view(-1,1)
            Tm = T[mask].view(-1,1)
            xs.append(Xm); ys.append(Ym); ts.append(Tm)
            count += Xm.shape[0]
    x = torch.cat(xs, 0)[:N]
    y = torch.cat(ys, 0)[:N]
    t = torch.cat(ts, 0)[:N]
    return x, y, t

def sample_ic(N):
    x, y, t = sample_outside(N)
    phi = compute_phi(x, y)
    u0 = torch.zeros_like(x)
    v0 = torch.zeros_like(x)
    return x, y, t, phi, u0, v0

def sample_collocation(N, delta=0.05):
    N_far  = int(0.8 * N)
    N_near = N - N_far

    xf, yf, tfar = sample_outside(N_far)
    phi_far = compute_phi(xf, yf)

    xs, ys, ts = [], [], []
    count = 0
    while count < N_near:
        m = N_near - count
        X = torch.rand(m,1, device=device)*(x_ub - x_lb) + x_lb
        Y = torch.rand(m,1, device=device)*(y_ub - y_lb) + y_lb
        T = torch.rand(m,1, device=device)*(t_ub - t_lb) + t_lb
        phi = compute_phi(X, Y)
        mask = phi.abs() < delta
        if mask.any():
            Xm = X[mask].view(-1,1)
            Ym = Y[mask].view(-1,1)
            Tm = T[mask].view(-1,1)
            xs.append(Xm); ys.append(Ym); ts.append(Tm)
            count += Xm.shape[0]

    x_near = torch.cat(xs, 0)[:N_near]
    y_near = torch.cat(ys, 0)[:N_near]
    t_near = torch.cat(ts, 0)[:N_near]
    phi_near = compute_phi(x_near, y_near)

    x = torch.cat([xf, x_near], dim=0)
    y = torch.cat([yf, y_near], dim=0)
    t = torch.cat([tfar, t_near], dim=0)
    phi = torch.cat([phi_far, phi_near], dim=0)
    return x, y, t, phi

def sample_walls(N):
    # bottom wall
    x1 = torch.rand(N,1, device=device)*(x_ub - x_lb) + x_lb
    y1 = torch.full((N,1), y_lb, device=device)
    t1 = torch.rand(N,1, device=device)*(t_ub - t_lb) + t_lb
    # top wall
    x2 = torch.rand(N,1, device=device)*(x_ub - x_lb) + x_lb
    y2 = torch.full((N,1), y_ub, device=device)
    t2 = torch.rand(N,1, device=device)*(t_ub - t_lb) + t_lb

    x = torch.cat([x1, x2], dim=0)
    y = torch.cat([y1, y2], dim=0)
    t = torch.cat([t1, t2], dim=0)
    phi = compute_phi(x, y)
    return x, y, t, phi

def sample_obstacle(N):
    xs, ys, ts = [], [], []
    count = 0
    while count < N:
        m = N - count
        X = torch.rand(m,1, device=device)*(x_ub - x_lb) + x_lb
        Y = torch.rand(m,1, device=device)*(y_ub - y_lb) + y_lb
        T = torch.rand(m,1, device=device)*(t_ub - t_lb) + t_lb
        mask = sphere_noslip_mask(X, Y)
        if mask.any():
            Xm = X[mask].view(-1,1)
            Ym = Y[mask].view(-1,1)
            Tm = T[mask].view(-1,1)
            xs.append(Xm); ys.append(Ym); ts.append(Tm)
            count += Xm.shape[0]
    x = torch.cat(xs, 0)[:N]
    y = torch.cat(ys, 0)[:N]
    t = torch.cat(ts, 0)[:N]
    phi = compute_phi(x, y)
    return x, y, t, phi

def sample_inlet(N):
    x   = torch.full((N,1), x_lb, device=device)
    y   = torch.rand(N,1, device=device)*(y_ub - y_lb) + y_lb
    t   = torch.rand(N,1, device=device)*(t_ub - t_lb) + t_lb
    phi = compute_phi(x, y)
    U_inf = hp.U_INLET
    Ut    = U_inf * (1 - torch.exp(-5*(t - t_lb)))
    Vt    = torch.zeros_like(Ut)
    return x, y, t, phi, Ut, Vt

def sample_outlet(N):
    x   = torch.full((N,1), x_ub, device=device)
    y   = torch.rand(N,1, device=device)*(y_ub - y_lb) + y_lb
    t   = torch.rand(N,1, device=device)*(t_ub - t_lb) + t_lb
    phi = compute_phi(x, y)
    return x, y, t, phi

# ── LSTM‐PINN model ─────────────────────────────────────────────────────────────

class LSTM_PINN_NS2D(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(4, hidden_size, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 3)

    def forward(self, x, y, t, phi):
        # ensure proper shape
        x   = x.view(-1,1); y   = y.view(-1,1)
        t   = t.view(-1,1); phi = phi.view(-1,1)
        seq = torch.cat([x, y, t, phi], dim=-1).unsqueeze(1)  # [B,1,4]
        h, _ = self.rnn(seq)                                  # [B,1,H]
        uvp  = self.fc(h[:, -1, :])                           # [B,3]
        return uvp[:,0:1], uvp[:,1:2], uvp[:,2:3]

model = LSTM_PINN_NS2D(hidden_size, num_layers).to(device)

# ── Physics residual & losses ─────────────────────────────────────────────────

def NS_res(x, y, t, phi):
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    phi = phi.clone().detach()

    u, v, p = model(x, y, t, phi)
    ones = torch.ones_like(u)

    u_t = torch.autograd.grad(u, t, grad_outputs=ones, create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=ones, create_graph=True)[0]

    u_x = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=ones, create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=ones, create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=ones, create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, grad_outputs=ones, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=ones, create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=ones, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=ones, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=ones, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=ones, create_graph=True)[0]

    cont = u_x + v_y
    ru   = u_t + (u*u_x + v*u_y) + p_x - nu*(u_xx + u_yy)
    rv   = v_t + (u*v_x + v*v_y) + p_y - nu*(v_xx + v_yy)

    return ru, rv, cont

mse = nn.MSELoss()

def compute_losses():
    x0, y0, t0, phi0, u0_tgt, v0_tgt = sample_ic(N_ic)
    u0, v0, _ = model(x0, y0, t0, phi0)
    L_ic = mse(u0, u0_tgt) + mse(v0, v0_tgt)

    xc, yc, tc, phi_c = sample_collocation(N_colloc)
    ru, rv, cont = NS_res(xc, yc, tc, phi_c)
    L_pde = mse(ru, torch.zeros_like(ru)) \
          + mse(rv, torch.zeros_like(rv)) \
          + mse(cont, torch.zeros_like(cont))

    xw, yw, tw, phi_w = sample_walls(N_walls)
    uw, vw, _ = model(xw, yw, tw, phi_w)
    L_walls = mse(uw, torch.zeros_like(uw)) + mse(vw, torch.zeros_like(vw))

    xo, yo, to, phi_o = sample_obstacle(N_obs)
    uo, vo, _ = model(xo, yo, to, phi_o)
    L_obs = mse(uo, torch.zeros_like(uo)) + mse(vo, torch.zeros_like(vo))

    xi, yi, ti, phi_i, Ut, Vt = sample_inlet(N_inlet)
    ui, vi, _ = model(xi, yi, ti, phi_i)
    L_inlet  = mse(ui, Ut) + mse(vi, Vt)

    xo2, yo2, to2, phi_o2 = sample_outlet(N_outlet)
    _, _, po = model(xo2, yo2, to2, phi_o2)
    L_outlet = mse(po, torch.zeros_like(po))

    L = (
        lambda_ic    * L_ic
      + lambda_pde   * L_pde
      + lambda_walls * L_walls
      + lambda_obs   * L_obs
      + lambda_inlet * L_inlet
      + lambda_outlet* L_outlet
    )
    return L, L_ic, L_pde, L_walls, L_obs, L_inlet, L_outlet

# ── Training ───────────────────────────────────────────────────────────────────

def train():
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    history = {k:[] for k in ["total","ic","pde","walls","obs","inlet","outlet"]}

    for ep in range(1, epochs+1):
        opt.zero_grad()
        losses = compute_losses()
        losses[0].backward()
        opt.step()
        for key, val in zip(history.keys(), losses):
            history[key].append(val.item())
        if ep % 20 == 0:
            print(f"Epoch {ep}/{epochs} | "
                  f"Total={losses[0]:.3e} IC={losses[1]:.3e} PDE={losses[2]:.3e} "
                  f"Walls={losses[3]:.3e} Obs={losses[4]:.3e} "
                  f"Inlet={losses[5]:.3e} Outlet={losses[6]:.3e}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/LSTM_NS2d_improved.pth")
    np.save("loss/loss_NS2d_improved.npy", history)
    print("✔ Model and loss history saved ✔")

if __name__ == "__main__":
    train()
