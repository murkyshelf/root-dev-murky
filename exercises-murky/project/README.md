# Physics-Informed Neural Network Spring Model with ROOT TMVA SOFIE

## Overview

This project demonstrates a complete **scientific machine learning pipeline** combining:

* Physics-informed neural networks (PINN)
* PyTorch training
* ONNX model export
* ROOT TMVA SOFIE inference

The goal of the model is to **predict the displacement of a spring over time** using a neural network trained with both data and physical constraints.

The trained neural network is exported to **ONNX format** and then executed inside **ROOT using TMVA SOFIE**, which enables fast C++ inference.

---

# Physical System

The system modeled is a **mass–spring oscillator** governed by classical mechanics.

Hooke’s law describes the restoring force of a spring:

$
F = -kx
$

Using Newton’s second law:

$
F = m\frac{d^2x}{dt^2}
$

Combining both equations gives the differential equation of motion:

$
m\frac{d^2x}{dt^2} + kx = 0
$

The analytical solution is:

$
x(t) = A\cos(\omega t) + B\sin(\omega t)
$

where

$
\omega = \sqrt{\frac{k}{m}}
$

The neural network is trained to approximate the function:

```
t → x(t)
```

---

# Project Structure

```
project/
│
├── spring_pinn.py
│   Physics-informed neural network training script
│
├── export_spring_model.py
│   Exports the trained model to ONNX format
│
├── spring_pinn_weights.pth
│   Saved neural network weights after training
│
├── spring_model.onnx
│   Exported ONNX model for inference
│
├── spring_sofie.C
│   ROOT macro that loads and runs the model using SOFIE
│
└── README.md
    Project documentation
```

---

# Training the Model

The model is implemented using **PyTorch**.

The network learns the displacement function:

```
x(t)
```

Training combines two loss terms:

### Data loss

The network output is compared against the analytical solution.

```
MSE(predicted displacement, true displacement)
```

### Physics loss

The model is constrained by the spring differential equation.

$
m\frac{d^2x}{dt^2} + kx = 0
$

Autograd is used to compute derivatives.

### Total loss

```
loss = data_loss + λ * physics_loss
```

---

# Running Training

```
python spring_pinn.py
```

The training script:

1. Builds the neural network
2. Generates training data
3. Enforces the physical constraint
4. Optimizes the model parameters
5. Saves the weights

Output:

```
spring_pinn_weights.pth
```

---

# Exporting the Model to ONNX

The trained model is converted into ONNX format.

ONNX (Open Neural Network Exchange) is a framework-independent representation of neural networks.

Export command:

```
python export_spring_model.py
```

The export process:

```
PyTorch model
      ↓
Trace computation graph
      ↓
Convert operators to ONNX
      ↓
Embed trained weights
      ↓
Write spring_model.onnx
```

Output:

```
spring_model.onnx
```

---

# Running the Model in ROOT

ROOT loads the ONNX model using **TMVA SOFIE**.

SOFIE parses the ONNX graph and executes the model inside ROOT.

Run the macro:

```
root spring_sofie.C
```

Example output:

```
Model loaded with SOFIE
t = 0.0  -> x = 1.00
t = 0.5  -> x = 0.87
t = 1.0  -> x = 0.54
```

This shows the neural network predicting the spring displacement.

---

# Full Pipeline

The complete workflow of this project is:

```
Physics equations
      ↓
PINN training (PyTorch)
      ↓
Saved weights
      ↓
ONNX export
      ↓
ROOT TMVA SOFIE parser
      ↓
C++ inference
```

This pipeline demonstrates how machine learning models trained in Python can be deployed in high-performance C++ environments.

---

# Technologies Used

* PyTorch (neural network training)
* ONNX (model exchange format)
* ROOT (scientific computing framework)
* TMVA (Toolkit for Multivariate Analysis)
* SOFIE (C++ inference engine)

Relevant frameworks:

* PyTorch
* Open Neural Network Exchange
* Toolkit for Multivariate Data Analysis

---

# Applications

This approach can be extended to more complex physical systems:

* damped spring oscillators
* mechanical vibration systems
* robotics joint dynamics
* control systems
* physics simulations

Physics-informed neural networks are widely used in **scientific machine learning**.

---

# Future Improvements

Possible extensions include:

* damping terms in the spring equation
* nonlinear spring models
* parameter estimation from experimental data
* real-time inference using generated C++ code

