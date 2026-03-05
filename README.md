# Physics-Informed Neural Network (PINN) for the 1D Heat Equation, short comprative study

## Overview
This project implements a Physics-Informed Neural Network (PINN) in PyTorch 
to solve the 1D heat equation. The network approximates the solution without 
labeled data by incorporating the governing PDE, initial condition, and 
boundary conditions into the loss function via automatic differentiation.

The project also conducts a comparative study of activation functions and 
optimizers to analyze their impact on solution accuracy and convergence.


## Problem Description

We solve the 1D heat equation:

u_t = α u_xx

on the domain x ∈ [0,1] and t ∈ [0,T].

Initial condition:

u(x,0) = sin(πx)

Boundary conditions:

- Left boundary: u(0,t) = 0
- Right boundary: u(1,t) = 0


## Method and Study Design

The PINN consists of a fully connected neural network with 3 hidden layers of 50 neurons each.  
This architecture was chosen to balance:

- Sufficient capacity to learn the smooth solution
- Reasonable training time
- Avoiding overfitting for a simple 1D PDE

### Hypothesis

- Smooth activations (Tanh, Sin) will approximate the PDE solution more accurately than ReLU because second derivatives are required for the heat equation.
- Additionally, Sin may perform slightly better than Tanh for this problem because the solution itself has sinusoidal behavior, so the network can represent it more naturally, leading to faster convergence and lower error.
- Adam optimizer will converge faster and more stably than SGD. We expect this as Adam adapts the learning rate for each parameter individually using estimates of first and second moments of the gradient, whereas SGD has a fixed learning rate for all parameters.

### Study Variables

- **Activation functions:** Tanh, Sin, ReLU  
- **Optimizers:** Adam, SGD  
- **Loss function:** PDE residual + Initial condition loss + Boundary condition loss  

### Evaluation Metrics

- Mean Squared Error (MSE) at t = 0.25, 0.5, 0.75, 1.0  
- Maximum absolute error across the spatial domain  
- Training convergence behavior over epochs  

This setup allows a focused study on activation functions and optimizers while keeping the network architecture fixed.


## Experiments

- Each combination of activation function and optimizer was trained for 5000 epochs.  
- Random collocation points were sampled for the PDE residual calculation.  
- Predictions were evaluated at fixed time slices (t=0,0.25,0.5,0.75,1.0) to compare against the analytical solution.  


## Results - DOUBLE CHECK

- **Activation functions:**  
  - Tanh and Sin achieved lower errors and smoother approximations of the PDE solution.  
  - ReLU performed worse due to its non-smooth nature, which limits accurate second derivatives.  

- **Optimizers:**  
  - Adam consistently converged faster than SGD and reached lower final losses.  

- **Visualization:**  
  - Convergence curves show differences in training stability across experiments.  
  - Heatmaps illustrate the temporal evolution of temperature along the rod, confirming the network captures the expected behavior.


## Conclusion - DOUBLE CHECK

The experiments confirm the hypothesis:

- Smooth activations (Tanh, Sin) are better suited for PDEs requiring higher-order derivatives.  
- Adam outperforms SGD in convergence speed.  

This study demonstrates that activation function and optimizer selection significantly affect PINN performance, even when network architecture is fixed.


## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt

2, Train the model
python train.py

3. Visualise Results
python visualise.py
