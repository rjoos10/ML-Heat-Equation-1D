import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# Setup
# =========================

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.makedirs("results", exist_ok=True)

# =========================
# Custom Activation
# =========================

class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

# =========================
# PINN Model
# =========================

class PINN(nn.Module):
    def __init__(self, layers, activation_fn):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = activation_fn

    def forward(self, x, t):
        X = torch.cat([x, t], dim=1)
        for layer in self.layers[:-1]:
            X = self.activation(layer(X))
        X = self.layers[-1](X)
        return X

# =========================
# Boundary Conditions
# =========================

def u_b_left(t):
    return torch.zeros_like(t)

def u_b_right(t):
    return torch.zeros_like(t)

# =========================
# PINN Loss
# =========================

def pinn_loss(model, x_colloc, t_colloc, x_i, t_i, u_i, x_b0, x_b1, t_b, alpha=1.0):
    x_colloc = x_colloc.clone().requires_grad_(True)
    t_colloc = t_colloc.clone().requires_grad_(True)
    u = model(x_colloc, t_colloc)
    u_t = torch.autograd.grad(u, t_colloc, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_colloc, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_colloc, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    loss_pde = torch.mean((u_t - alpha * u_xx)**2)
    u_ic = model(x_i, t_i)
    loss_ic = torch.mean((u_ic - u_i)**2)
    u_b0_pred = model(x_b0, t_b)
    u_b1_pred = model(x_b1, t_b)
    u_b0_true = u_b_left(t_b)
    u_b1_true = u_b_right(t_b)
    loss_bc = torch.mean((u_b0_pred - u_b0_true)**2) + torch.mean((u_b1_pred - u_b1_true)**2)
    return loss_pde + loss_ic + loss_bc

# =========================
# Main Experiment
# =========================

def main():

    # Training data
    N_colloc = 200
    x_colloc = torch.rand(N_colloc,1,device=device)
    t_colloc = torch.rand(N_colloc,1,device=device)
    N_ic = 200
    x_i = torch.rand(N_ic,1,device=device)
    t_i = torch.zeros_like(x_i)
    u_i = torch.sin(np.pi*x_i).to(device)
    x_b0 = torch.zeros_like(x_i)
    x_b1 = torch.ones_like(x_i)
    t_b = t_i.clone()

    # Experiment configurations
    optimizers = {
        'Adam': lambda params: optim.Adam(params, lr=1e-3),
        'SGD': lambda params: optim.SGD(params, lr=1e-3),
    }

    activations = {
        'tanh': nn.Tanh(),
        'sin': SinActivation(),
        'ReLU': nn.ReLU()
    }

    results = {}
    trained_models = {}

    # Training loop
    for act_name, act_fn in activations.items():
        for opt_name, opt_fn in optimizers.items():
            print(f'Training with {act_name} activation and {opt_name} optimizer')

            model = PINN([2,50,50,50,1], act_fn).to(device)
            optimizer = opt_fn(model.parameters())
            loss_history = []

            for epoch in range(5000):
                optimizer.zero_grad()
                loss = pinn_loss(model, x_colloc, t_colloc, x_i, t_i, u_i, x_b0, x_b1, t_b)
                loss.backward()
                optimizer.step()
                loss_history.append(loss.item())

            # Evaluation
            x_test = torch.linspace(0,1,100).view(-1,1).to(device)
            t_slices = torch.linspace(0,1,5).view(-1,1).to(device)
            mse_list = []
            max_err_list = []

            for t_val in t_slices:
                t_grid = t_val.repeat(len(x_test),1)
                u_pred = model(x_test, t_grid).detach().cpu().numpy()
                u_true = np.exp(-np.pi**2 * t_val.cpu().numpy()) * np.sin(np.pi*x_test.cpu().numpy())
                mse_list.append(np.mean((u_pred - u_true)**2))
                max_err_list.append(np.max(np.abs(u_pred - u_true)))

            results[(act_name,opt_name)] = {
                'mse': np.mean(mse_list),
                'max_err': np.max(max_err_list),
                'loss_history': loss_history
            }
            trained_models[(act_name,opt_name)] = model

            # Save model
            torch.save(model.state_dict(), f"results/model_{act_name}_{opt_name}.pt")

            print(f"MSE: {np.mean(mse_list):.4e} | Max Error: {np.max(max_err_list):.4e}")

    print("Experiment completed")

    # =========================
    # Convergence Plot
    # =========================

    plt.figure(figsize=(12,6))
    for (act_name, opt_name), res in results.items():
        plt.plot(res['loss_history'], label=f'{act_name}-{opt_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('PINN Training Convergence')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig("results/convergence.png")
    plt.show()

    # =========================
    # Heatmaps & Error Comparisons
    # =========================

    for (act_name, opt_name), model in trained_models.items():
        model.eval()
        x_vals = torch.linspace(0,1,100).view(-1,1).to(device)
        t_vals = torch.linspace(0,1,100).view(-1,1).to(device)
        U = np.zeros((len(t_vals), len(x_vals)))

        for i, t in enumerate(t_vals):
            t_grid = t.repeat(len(x_vals),1)
            u_pred = model(x_vals, t_grid).detach().cpu().numpy()
            U[i,:] = u_pred[:,0]

        plt.figure(figsize=(8,6))
        plt.imshow(U, extent=[0,1,0,1], origin='lower', aspect='auto', cmap='hot')
        plt.colorbar(label='u(x,t)')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title(f'Heatmap: {act_name} activation + {opt_name} optimizer')
        plt.savefig(f"results/heatmap_{act_name}_{opt_name}.png")
        plt.show()

    # MSE Comparison
    activ_opt_labels = [f'{act}-{opt}' for (act,opt) in results.keys()]
    mse_values = [res['mse'] for res in results.values()]
    max_err_values = [res['max_err'] for res in results.values()]

    plt.figure(figsize=(10,5))
    plt.bar(activ_opt_labels, mse_values, color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel('MSE (averaged over time slices)')
    plt.title('MSE Comparison Across Activation Functions and Optimizers')
    plt.grid(axis='y')
    plt.savefig("results/mse_comparison.png")
    plt.show()

    # Max Error Comparison
    plt.figure(figsize=(10,5))
    plt.bar(activ_opt_labels, max_err_values, color='salmon')
    plt.xticks(rotation=45)
    plt.ylabel('Max Error (over time slices)')
    plt.title('Max Error Comparison Across Activation Functions and Optimizers')
    plt.grid(axis='y')
    plt.savefig("results/max_error_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()

