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

- Mean Squared Error (MSE) averaged over the times t = 0.25, 0.5, 0.75, 1.0  
- Maximum absolute error across the spatial domain averaged over the times t = 0.25, 0.5, 0.75, 1.0 
- Training convergence behavior over epochs  

This setup allows a focused study on activation functions and optimizers while keeping the network architecture fixed.


## Experiments

- Each combination of activation function and optimizer was trained for 5000 epochs.  
- Random collocation points were sampled for the PDE residual calculation.  
- Predictions were evaluated at fixed time slices (t=0,0.25,0.5,0.75,1.0) to compare against the analytical solution.  


## Results

- **Activation Functions:**  
  - Tanh generally achieves slightly lower MSE and Max Error compared to Sin, especially with the Adam optimizer, likely due to smoother derivatives better capturing the PDE solution.  
  - Tanh and Sin show similar total loss trends over time with the same optimizer, while ReLU shows lower total loss.
  - Max Error and MSE are comparable between Tanh and Sin, but Sin tends to have slightly larger errors for both Adam and SGD optimizers. ReLU appears numerically strong but is not reliable due to non-physical behavior shown in the heatmap results.

- **Optimizers:**  
  - Adam achieves lower total loss compared to SGD. However, sudden peaks are present, likely due to its adaptive learning rate and gradient-based updates.
  - SGD converges faster initially but stabilizes at higher loss values, indicating less accurate solutions.
  - In all cases the Adam optimizer gives a higher Max error and MSE value compared to SGD.
  - These results show that optimizer choice significantly impacts PINN performance, sometimes more than activation function choice.

- **Visualization:**  
  - Convergence curves differentiate training stability:  
    - SGD converges early but to higher loss values.  
    - Adam reaches lower loss but may show mild oscillations or peaks.  
    - ReLU shows low numerical loss but fails physically when looking at the heatmaps.
  - Heatmaps confirm Tanh and Sin with the Adam optimizer correctly capture the temporal evolution of temperature along the rod.  
  - SGD produces asymmetries in the solution, highlighting its limitations, despite lower MSE and Max error values.  
  - ReLU heatmaps do not match expected physical behavior, so despite low Max Error or MSE and total loss, it is not suitable for this PDE.

## Conclusion

- **Activation functions:** Smooth activations (Tanh, Sin) are better suited for PDEs requiring accurate higher-order derivatives. Tanh slightly outperforms Sin in MSE, Max Error, and total loss when combined with Adam.  
- **Optimizer choice:** Adam provides more stable training and lower total loss. SGD converges early but often to less accurate solutions.  
- **Numerical metrics vs. physical realism:** ReLU may produce low Max Error or MSE but fails to capture the correct solution, emphasizing the need for visual evaluation.  
- **Overall insight:** Both activation function and optimizer significantly affect PINN performance. For the heat equation problem studied, Tanh with the Adam optimizer provides the most accurate and physically realistic solution.

