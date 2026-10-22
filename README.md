# Deep Learning-Based Nonlinear Electrode Actuation Modeling

## Overview

This project investigates the use of PyTorch neural networks to predict and optimize the deformation of a simulated seven-electrode actuator.

The simulation models two challenges relevant to adaptive optical systems:

- Nonlinear voltage-dependent actuator response.
- Crosstalk between neighboring electrode regions.

**Disclaimer:** This is an independent educational prototype using synthetic data. It does not use experimental data from or reproduce the physical characteristics of a specific laboratory device.

## Methodology

A synthetic dataset of 10,000 voltage–deformation pairs was generated using a simplified nonlinear actuator model.

The dataset includes randomly actuated, single-electrode, and two-electrode configurations.

A multilayer perceptron (MLP) was implemented in PyTorch:

- Input: 7 normalized electrode voltages
- Hidden layers: 64 and 64 neurons with ReLU activation
- Output: 7 regional deformation predictions
- Loss function: Mean Squared Error
- Optimizer: Adam

## Results

The trained MLP achieved a test RMSE of **0.006696** on the revised synthetic dataset.

In an earlier benchmark using randomly sampled configurations, the models achieved:

| Model | Test RMSE |
|---|---:|
| Linear Regression | 0.034157 |
| kNN Regression | 0.063392 |
| PyTorch MLP | 0.006843 |

These results are specific to the simulated dataset and evaluated model configurations.

Targeted training samples substantially improved the MLP's ability to predict isolated-electrode actuation and neighboring-region coupling.

## Inverse Optimization

A trained forward neural network was used as a differentiable surrogate model.

Electrode voltages were optimized using PyTorch automatic differentiation to reproduce a desired deformation profile.

The resulting voltage configuration achieved a physical-model RMSE of **0.004589** on one example target.

This demonstrates the feasibility of the approach within the simplified simulation, rather than validated performance on experimental hardware.

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the dataset:

```bash
python data/generate_data.py
```

Train the neural network:

```bash
python train.py
```

Evaluate the models:

```bash
python compare_models.py
python crosstalk_analysis.py
```

Perform inverse optimization:

```bash
python inverse_optimization.py
```

## Future Work

- Evaluate additional voltage configurations.
- Improve the physical plausibility of the synthetic actuator model.
- Investigate wavefront phase-map prediction.
- Compare ML-based inverse optimization against conventional numerical optimization.
- Explore model performance on experimental data when available.

## Tools

Python, PyTorch, NumPy, scikit-learn, SciPy, and Matplotlib.