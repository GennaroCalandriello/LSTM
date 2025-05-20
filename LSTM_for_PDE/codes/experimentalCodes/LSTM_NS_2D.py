import os
import torch
import torch.nn as nn
import numpy as np
import hyperpar as hp

# 0) Disable CuDNN for double‐backward
torch.backends.cudnn.enabled = False

initialCondition = ["TaylorGreen", "Gaussian"]
indexIC = 2

want_obstacle = True
# 1) Hyperparameters & device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nu = hp.NU  # kinematic viscosity
rho = getattr(hp, 'RHO', 1.0)
hidden_size = hp.HIDDEN_SIZE
num_layers = hp.NUM_LAYERS
lr = hp.LR
epochs = 1000
N_ic = hp.N_IC
N_colloc = hp.N_COLLOCATION
N_bc = hp.N_BC
N_obs = hp.N_OBSTACLE  # # obstacle‐condition points
input_size = 3  # (x,y,t)

# Domain bounds
x_lb, x_ub, y_lb, y_ub = hp.X_LB, hp.X_UB, hp.Y_LB, hp.Y_UB
t_lb, t_ub = hp.T_LB, hp.T_UB

# Cylinder parameters
cx, cy, r = hp.cx, hp.cy, hp.r

# 2) Taylor–Green analytic solution
def initial_conditions(x, y, t):
    """Taylor–Green vortex initial conditions"""
    if indexIC == 0:
            
        exp2 = torch.exp(-2 * nu * t)
        exp4 = torch.exp(-4 * nu * t)
        u = torch.cos(x) * torch.sin(y)*exp2
        v = -torch.sin(x) * torch.cos(y)*exp2
        p = -rho / 4 * (torch.cos(2*x) + torch.cos(2*y))*exp4
    
    if indexIC == 1:
        u = torch.exp(-((x - cx)**2 + (y - cy)**2) / (2 * r**2)) * torch.cos(x) * torch.sin(y)
        v = torch.exp(-((x - cx)**2 + (y - cy)**2) / (2 * r**2)) * -torch.sin(x) * torch.cos(y)
        p = -rho / 4 * (torch.cos(2*x) + torch.cos(2*y))
    # base: zero everywhere
    u = torch.zeros_like(x)
    v = torch.zeros_like(x)
    p = torch.zeros_like(x)

    # only apply inlet profile at t=0
    if indexIC == 2 and torch.allclose(t, torch.zeros_like(t)):
        # inlet stripe (on the left wall) for y in [y_min,y_max]
        y_min, y_max = 0.2, 0.8
        U0 = 0.6
        # define a small thickness delta over which x transitions
        delta = 0.05*(x_ub - x_lb)

        # mask for “inlet region”: x within [x_lb, x_lb+delta] AND y in stripe
        inlet_mask = (
            (x >= x_lb) & (x <= x_lb + delta) &
            (y >= y_min) & (y <= y_max)
        )
        u[inlet_mask] = U0
        v[inlet_mask] = 0.0
        # pressure can stay zero (it will adjust via PINN)
        p[inlet_mask] = 0.0
         
    return u, v, p

def circle_mask(x, y, xc, yc, r):
    """Circle mask for obstacle"""
    
    return (x-xc)**2 + (y-yc)**2 < r**2

# 3) Sampling
# 3.1) Initial condition: interior, t=0
x_ic = torch.rand(N_ic, 1)*(x_ub - x_lb) + x_lb
y_ic = torch.rand(N_ic, 1)*(y_ub - y_lb) + y_lb
t_ic = torch.zeros_like(x_ic)

mask_ic = ~circle_mask(x_ic, y_ic, cx, cy, r)
x_ic, y_ic, t_ic = [t[mask_ic].reshape(-1, 1) for t in (x_ic, y_ic, t_ic)]
u_ic, v_ic, p_ic = initial_conditions(x_ic, y_ic, t_ic)


# 3.2) Collocation points (interior)
x_coll = torch.rand(N_colloc,1)*(x_ub-x_lb) + x_lb
y_coll = torch.rand(N_colloc,1)*(y_ub-y_lb) + y_lb
t_coll = torch.rand(N_colloc,1)*(t_ub-t_lb) + t_lb

# Remove points inside the circle
mask_coll = ~circle_mask(x_coll, y_coll, cx, cy, r)
#zero around the circle, IC does respect the BC at time t=0
x_coll = x_coll[mask_coll].reshape(-1,1)
y_coll = y_coll[mask_coll].reshape(-1,1)
t_coll = t_coll[mask_coll].reshape(-1,1)

x_ic, y_ic, t_ic, u_ic, v_ic, p_ic = [t.to(device) for t in (x_ic, y_ic, t_ic, u_ic, v_ic, p_ic)]
x_coll, y_coll, t_coll = [t.to(device) for t in (x_coll, y_coll, t_coll)]


def sample_boundary(N):
    theta = 2*torch.pi*torch.rand(N,1)
    x_b = cx + r * torch.cos(theta)
    y_b = cy + r * torch.sin(theta)
    t_b = torch.rand(N,1)*(t_ub-t_lb) + t_lb
    
    return [f.to(device).requires_grad_() for f in (x_b, y_b, t_b)]

# 4) Define LSTM‐PINN model
class LSTM_PINN_NS2D(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 3),
            nn.Tanh()
        )
        
    def forward(self, x, y, t):
        seq = torch.cat([x,y,t], dim=-1).unsqueeze(1)
        h, _ = self.rnn(seq)
        uvp = self.fc(h[:, -1])
        return uvp[:,0:1], uvp[:,1:2], uvp[:,2:3]

model = LSTM_PINN_NS2D(input_size, hidden_size, num_layers).to(device)

def inlet_velocity():
    # define your stripe in y
    y_min, y_max = 0.2, 2
    U0 = 2

    # sample more than you need
    x_in = torch.full((N_bc,1), x_lb,
                    dtype=torch.float32, device=device)
    y_in = torch.rand(N_bc,1, device=device)*(y_ub-y_lb) + y_lb
    t_in = torch.rand(N_bc,1, device=device)*(t_ub-t_lb) + t_lb

    # keep only the stripe + outside obstacle
    mask_stripe = (y_in >= y_min) & (y_in <= y_max)
    mask_fluid  = ~circle_mask(x_in, y_in, cx, cy, r)
    mask_in     = (mask_stripe & mask_fluid).squeeze()

    x_in = x_in[mask_in].reshape(-1,1)
    y_in = y_in[mask_in].reshape(-1,1)
    t_in = t_in[mask_in].reshape(-1,1)

    u_in = torch.full((x_in.shape[0],1), U0,
                    dtype=torch.float32, device=device)
    v_in = torch.zeros_like(u_in)
    return x_in, y_in, t_in, u_in, v_in

# 5) PDE residuals
def NS_res(x,y,t):
    um, vm = 0.5, 0.5
    for g in (x,y,t): g.requires_grad_(True)
    
    u,v,p = model(x,y,t)
    
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
    
    continuity = u_x + v_y
    pu = u_t + (u*u_x + v*u_y) + p_x - nu*(u_xx+u_yy)
    pv = v_t + (u*v_x + v*v_y) + p_y - nu*(v_xx+v_yy)
    entropy = (u-um)*u + (v-vm)*v 
    
    return pu, pv, continuity, entropy

# 6) Loss
mse = nn.MSELoss()
def loss_functions():
    # IC loss vs analytic
    u0, v0, p0 = model(x_ic, y_ic, t_ic)
    L_ic = mse(u0, u_ic) + mse(v0, v_ic) 
    # PDE residual loss
    pu,pv,cont = NS_res(x_coll, y_coll, t_coll)
    L_pde = mse(pu, torch.zeros_like(pu)) + mse(pv, torch.zeros_like(pv)) + mse(cont, torch.zeros_like(cont))
   # BC loss
    x_bc, y_bc, t_bc = sample_boundary(N_bc)
    u_bc, v_bc, p_bc = model(x_bc, y_bc, t_bc)
    L_bc = mse(u_bc, u_bc*0) + mse(v_bc, v_bc*0) 
    L_tot = hp.LAMBDA_DATA*L_ic + hp.LAMBDA_PDE*L_pde + hp.LAMBDA_BC*L_bc
    return L_tot, L_ic, L_pde, L_bc

def loss_functions2():
    # 1) Initial‐condition loss (analytic)
    u0_pred, v0_pred, _     = model(x_ic,  y_ic,  t_ic)
    L_ic = mse(u0_pred, u_ic) + mse(v0_pred, v_ic)

    # 2) PDE residual + entropy loss
    pu, pv, cont, entropy   = NS_res(x_coll, y_coll, t_coll)
    L_pde     = (
          mse(pu,   torch.zeros_like(pu))
        + mse(pv,   torch.zeros_like(pv))
        + mse(cont, torch.zeros_like(cont))
    )
    L_entropy = mse(entropy, torch.zeros_like(entropy))

    # 3) No‐slip on cylinder
    x_b, y_b, t_b            = sample_boundary(N_bc)
    u_b_pred, v_b_pred, _    = model(x_b, y_b, t_b)
    L_obs     = (
          mse(u_b_pred, torch.zeros_like(u_b_pred))
        + mse(v_b_pred, torch.zeros_like(v_b_pred))
    )

    # 4) Inlet BC at x = x_lb, u = U0, v = 0
    x_in, y_in, t_in, u_in, v_in = inlet_velocity()
    u_in_pred, v_in_pred, _       = model(x_in, y_in, t_in)
    # L_in      = (
    #       mse(u_in_pred, u_in)
    #     + mse(v_in_pred, v_in)
    # )

    # 5) Outlet zero-gradient BC at x = x_ub: ∂u/∂x = 0
    x_out = torch.full((N_bc,1), x_ub, dtype=torch.float32, device=device)
    y_out = torch.rand(N_bc,1, device=device)*(y_ub-y_lb) + y_lb
    t_out = torch.rand(N_bc,1, device=device)*(t_ub-t_lb) + t_lb
    x_out.requires_grad_()
    u_out_pred, _, _ = model(x_out, y_out, t_out)
    u_x_out = torch.autograd.grad(
        u_out_pred, x_out,
        grad_outputs=torch.ones_like(u_out_pred),
        create_graph=True
    )[0]
    L_out     = mse(u_x_out, torch.zeros_like(u_x_out))

    # 6) Pressure gauge at outlet to fix constant offset
    x_ref = torch.tensor([[x_ub]], dtype=torch.float32, device=device)
    y_ref = torch.tensor([[cy]],   dtype=torch.float32, device=device)
    t_ref = torch.zeros((1,1),     dtype=torch.float32, device=device)
    _, _, p_ref             = model(x_ref, y_ref, t_ref)
    L_p     = mse(p_ref, torch.zeros_like(p_ref))

    # 7) Combine boundary‐condition losses
    L_bc    = L_obs + L_out + L_p

    # 8) Total loss
    L_tot = (
          hp.LAMBDA_DATA    * L_ic
        + hp.LAMBDA_PDE     * L_pde
        + hp.LAMBDA_ENTROPY * L_entropy
        + hp.LAMBDA_BC      * L_bc
    )

    return L_tot, L_ic, L_pde, L_bc, L_entropy

# 7) Training loop
def train_noBatches():
    history={
        "total_loss": [],
        "pde_loss": [],
        "bc_loss": [],
        "ic_loss": [],
        "obs_loss": []}
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    for ep in range(1, epochs+1):
        opt.zero_grad()
        L, L_ic, L_pde, L_bc, L_entropy = loss_functions2()
        L.backward()
        opt.step()
        if ep % 10 == 0:
            print(f"Ep {ep}/{epochs}: L={L.item():.2e}, IC={L_ic.item():.2e}, PDE={L_pde.item():.2e}, BC={L_bc.item():.2e}, Entropy={L_entropy.item():.2e}")
        # registro le losses
        history["total_loss"].append(L.item())
        history["pde_loss"].append(L_pde.item())
        history["bc_loss"].append(L_bc.item())
        history["ic_loss"].append(L_ic.item())
    # save
    # save model and los
    os.makedirs('models', exist_ok=True)
    os.makedirs("loss", exist_ok=True)
    if hp.SAVE_MODEL:
        try:
            os.remove("models/LSTM_NS2d_tg_cyl.pth")
        except OSError:
            pass
        torch.save(model.state_dict(), "models/LSTM_NS2d_tg_cyl.pth")
        print("Model saved to models/LSTM_NS2d_tg_cyl.pth")
    if hp.SAVE_LOSS:
        try:
            os.remove("models/LSTM_NS2d_tg_cyl_loss.npy")
        except OSError:
            pass
        np.save("models/LSTM_NS2d_tg_cyl_loss.npy", history)
        print(" Loss saved to models/LSTM_NS2d_tg_cyl_loss.npy")

if __name__ == '__main__':
    
    train_noBatches()
    # train()
