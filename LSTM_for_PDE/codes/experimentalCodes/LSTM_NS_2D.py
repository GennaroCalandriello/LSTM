import os
import torch
import torch.nn as nn
import numpy as np
import hyperpar as hp

# 0) Disable CuDNN for double‐backward
torch.backends.cudnn.enabled = False

# 1) Hyperparameters & device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nu = hp.NU  # kinematic viscosity
hidden_size = hp.HIDDEN_SIZE
num_layers = hp.NUM_LAYERS
lr = hp.LR
epochs = hp.EPOCHS
N_ic = hp.N_IC  # # initial‐condition points
N_colloc = hp.N_COLLOCATION
N_bc = hp.N_BC  # # boundary‐condition points
N_obs = hp.N_OBSTACLE  # # obstacle points
lambda_ic = hp.LAMBDA_DATA
lambda_pde = hp.LAMBDA_PDE
lambda_bc = hp.LAMBDA_BC
lambda_obs = hp.LAMBDA_OBS
input_size = 3  # input size (x,y,z,t)

#Domain
x_lb, x_ub, y_lb, y_ub = hp.X_LB, hp.X_UB, hp.Y_LB, hp.Y_UB
t_lb, t_ub = hp.T_LB, hp.T_UB

# 2) Mask on obstacle
def sphere_noslip(x, y, xc =0.5, yc =0, r = 0.4):
    return ((x - xc) ** 2 + (y - yc) ** 2 ) < r**2

# 3) Sampling
# 3.1) IC
x_ic = torch.rand(N_ic, 1)*(x_ub -x_lb) + x_lb
y_ic = torch.rand(N_ic, 1)*(y_ub-y_lb) +y_lb
t_ic = torch.zeros_like(x_ic)

mask_ic = ~sphere_noslip(x_ic, y_ic)
x_ic, y_ic, t_ic = [v[mask_ic].reshape(-1, 1) for v in (x_ic, y_ic, t_ic)]
u_ic = torch.zeros_like(x_ic)
v_ic = torch.zeros_like(x_ic)

#device
x_ic, y_ic, t_ic, u_ic, v_ic = [n.to(device) for n in (x_ic, y_ic, t_ic, u_ic, v_ic)]

# 3.2) Collocation points
x_coll = torch.rand(N_colloc, 1)*(x_ub-x_lb) +x_lb
y_coll = torch.rand(N_colloc, 1)*(y_ub-y_lb) + y_lb
t_coll = torch.rand(N_colloc, 1) * (t_ub-t_lb) + t_lb

mask_coll = ~sphere_noslip(x_coll, y_coll)
x_coll, y_coll, t_coll = [v[mask_coll].reshape(-1, 1) for v in (x_coll, y_coll, t_coll)]

#device
x_coll, y_coll, t_coll = [n.to(device) for n in (x_coll, y_coll, t_coll)]

def boundary(N):
    x = (x_ub-x_lb) * torch.rand(N, 1) + x_lb
    y = (y_ub-y_lb) * torch.rand(N, 1) + y_lb
    mask_noslip = sphere_noslip(x, y)
    x_bc = x[mask_noslip].reshape(-1, 1)
    y_bc = y[mask_noslip].reshape(-1, 1)
    t_bc = (t_ub-t_lb) * torch.rand(x_bc.shape[0], 1) + t_lb
    return [f.to(device) for f in (x_bc, y_bc, t_bc)]
    
x_bc, y_bc, t_bc = boundary(N_bc)

# 4) definisco il modello LSTM-PINN

class LSTM_PINN_NS2D(nn.Module):
    #__init__ is the constructor, called when a new instance of the class is created
    def __init__(self, input_size, hidden_size, num_layers):
        #calls the constructor of the parent class (nn.Module)
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)
    
    def forward(self, x, y, t):
        seq = torch.cat([x, y, t], dim = -1).unsqueeze(1)
        h, _ = self.rnn(seq)
        #qui definisco i 3 vettori velocità (RANS) e la pressione
        uvp = self.fc(h[:, -1, :])
        return uvp[:, 0:1], uvp[:, 1:2], uvp[:, 2:3]

#5) inizializzo il modello
model = LSTM_PINN_NS2D(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers).to(device)
# model = LSTM_PINN_NS3D().to(device)
# 6) residui, qui sta la fisica!

def inlet_velocity_field(N):
    """Prescrive un campo di velocità uniforme in x, zero in y e z"""
    x_in = torch.full((N,1), hp.x_lb_inlet)
    y_in = torch.rand(N, 1)*(y_ub-y_lb) +y_lb
    t_in = torch.rand(N, 1)*(t_ub-t_lb) +t_lb
    
    return [v.to(device) for v in (x_in, y_in, t_in)]
    
    
def NS_res(x,y,t):
    
    #require gradient
    for g in (x,y,t):
        g.requires_grad_(True)
        
    u,v,p = model(x,y,t)
    ones = torch.ones_like(u)
    
    #time derivatives
    u_t = torch.autograd.grad(u, t, grad_outputs = ones, create_graph = True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=ones, create_graph=True)[0]

    
    #spatial derivatives
    u_x = torch.autograd.grad(u, x, grad_outputs= ones, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs= ones, create_graph=True)[0]

    
    v_x = torch.autograd.grad(v, x, grad_outputs= ones, create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs= ones, create_graph=True)[0]

    
    p_x = torch.autograd.grad(p, x, grad_outputs= ones, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs= ones, create_graph=True)[0]

    
    #Laplacian
    #laplaciano: xx components
    u_xx = torch.autograd.grad(u_x, x, grad_outputs= ones, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs= ones, create_graph=True)[0]

    
    #laplaciano: yy components
    u_yy = torch.autograd.grad(u_y, y, grad_outputs= ones, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs= ones, create_graph=True)[0]

    
    #equazione di continuità
    continuity = u_x + v_y
    
    #momentum equations
    p_u = u_t + (u*u_x + v*u_y ) + p_x -nu*(u_xx + u_yy )
    p_v = v_t + (u*v_x +v*v_y) + p_y -nu*(v_xx +v_yy)

    
    return p_u, p_v, continuity

# 7) Loss function
mse = nn.MSELoss()
def loss_functions():
    """Funzione di loss, che calcola la loss totale e le singole componenti"""
    """Tutte le parti del dominio devono essere campionate dal modello"""
    
    #Initia condition loss
    u0, v0, _ = model(x_ic, y_ic, t_ic)
    #MSE := 1/N SUM ((u0-u_ic)^2 + (v0-v_ic)^2 + (w0-w_ic)^2)
    L_ic = mse(u0, u_ic) + mse(v0, v_ic)
    
    #PDE collocation loss
    p_u, p_v, continuity = NS_res(x_coll, y_coll, t_coll)
    #Forcing the residuals to be zero, mse(p_i, 0*p_i), ma non so se è la soluzione migliore
    L_pde = mse(p_u, 0*p_u) +mse(p_v, 0*p_v)  +mse(continuity, 0*continuity)
    
    #BC loss no-slip
    u_bc, v_bc, _ = model(x_bc, y_bc, t_bc)
    L_bc = mse(u_bc, 0*u_bc) + mse(v_bc, 0*v_bc)
    
    #BC loss inlet
    x_in, y_in, t_in = inlet_velocity_field(hp.N_INLET)
    u_in_pred, v_in_pred, _ = model(x_in, y_in, t_in)
    
    #prescrivo una velocità uniforme U_INLET in x, zero in y e z
    U_target = torch.full_like(u_in_pred, hp.U_INLET)
    V_target = torch.zeros_like(v_in_pred)
    L_inlet = (mse(u_in_pred, U_target) +mse(v_in_pred, V_target))
    
    #Total loss
    L = lambda_ic*L_ic + lambda_pde*L_pde+lambda_obs*L_bc + hp.LAMBDA_INLET*L_inlet
    
    return L, L_ic, L_pde, L_bc, L_inlet

# 8) training function
def train_NS():
    # torch.cuda.empty_cache()
    opt = torch.optim.Adam(model.parameters(), lr = lr)
    history = {"total":[], "ic":[], "pde":[], "bc":[], "inlet":[]}
    for ep in range(1, epochs+1):
        opt.zero_grad()
        L, L_ic, L_pde, L_bc, L_inlet = loss_functions()
        L.backward()
        opt.step()
        history["total"].append(L.item())
        history["ic"].append(L_ic.item())
        history["pde"].append(L_pde.item())
        history["bc"].append(L_bc.item())
        history["inlet"].append(L_inlet.item())
        if ep % 20 == 0:
            print(f"Epoch {ep}/{epochs}, Loss: {L.item():.4e}, IC Loss: {L_ic.item():.4e}, PDE Loss: {L_pde.item():.4e}, BC Loss: {L_bc.item():.4e}, Inlet Loss: {L_inlet.item():.4e}")
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("loss", exist_ok=True)
    
    #salva il modello
    torch.save(model.state_dict(), "models/LSTM_NS2d.pth")
    print("♪♪♪ Model saved ♪♪♪")
    np.save("loss/loss_NS2d.npy", history)
    print("©©© Loss saved ©©©")
    
if __name__ == "__main__":
    train_NS()
        