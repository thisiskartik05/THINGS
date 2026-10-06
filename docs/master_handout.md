# Comprehensive Master Handout: Data-Driven Modeling of Galactic Rotation Curves

> **Paper**: *Data-driven modeling of rotation curves with artificial neural networks*  
> **Authors**: Gabriela Garcia-Arroyo, Isidro Gómez-Vargas, J. Alberto Vázquez (arXiv:2404.05833v2)  
> **Repository**: [https://github.com/thisiskartik05/THINGS](https://github.com/thisiskartik05/THINGS)  
> **Target Audience**: Engineering Physics Students & Astrophysical ML Researchers  

---

## 1. Astrophysics & Kinematic Foundations

### 1.1 The Circular Velocity Paradigm
In spiral galaxies, matter orbits in near-circular trajectories within the galactic plane. Equilibrium between centripetal acceleration and gravitational attraction dictates:

$$\frac{V^2(r)}{r} = -\frac{\partial \Phi(r)}{\partial r} = \frac{G M(<r)}{r^2}$$

where $M(<r)$ is the total enclosed mass within radius $r$.

### 1.2 Quadrature Mass Decomposition
The observed total rotational velocity $V(r)$ is decomposed into individual mass components:

$$V_{\text{total}}^2(r) = V_{\text{gas}}^2(r) + \Upsilon_\star V_\star^2(r) + V_{\text{halo}}^2(r)$$

1. **Neutral Gas ($V_{\text{gas}}$)**: Derived from $21\,\text{cm}$ HI emission maps (THINGS survey). Corrected by a factor of $1.4\times$ to account for primordial helium and metals.
2. **Stellar Component ($V_\star$)**: Observed at $3.6\,\mu\text{m}$ (Spitzer SINGS survey) where extinction is minimal. The surface density profile is modeled as an exponential disk:
   $$\Sigma(r) = \Upsilon_\star^{3.6} I_0^{3.6} \exp\left(-\frac{r}{R_d}\right)$$
   The parameter $\Upsilon_\star$ (stellar mass-to-light ratio) scales $V_\star^2(r)$. Two main assumptions are evaluated:
   - $\Upsilon_\star^{\text{fix}}$: Fixed to a theoretical value derived from a "diet Salpeter" Initial Mass Function (IMF).
   - $\Upsilon_\star^{\text{free}}$: Treated as a free parameter fit alongside the Dark Matter halo.
3. **Dark Matter Halo ($V_{\text{halo}}$)**: Modeled using the Navarro-Frenk-White (NFW) profile:
   $$\rho_{\text{NFW}}(r) = \frac{\rho_i}{\left(\frac{r}{R_s}\right) \left(1 + \frac{r}{R_s}\right)^2}$$
   The enclosed NFW mass $M_{\text{NFW}}(r)$ yields circular velocity $V_{\text{halo}}(r) = \sqrt{\frac{G M_{\text{NFW}}(r)}{r}}$.

---

## 2. Mathematical Intuition & Machine Learning Formulation

### 2.1 Non-Parametric Modeling vs Interpolation
Traditional interpolation methods like LOESS (Locally Weighted Scatterplot Smoothing) fit local polynomials piecewise, making them sensitive to bandwidth choice and noise. In contrast, an **Artificial Neural Network (ANN)** fits a **global, continuous non-linear mapping**:

$$f_\mathbf{\theta}: r \longrightarrow \begin{bmatrix} V(r) \\ \sigma_V(r) \end{bmatrix}$$

where $\mathbf{\theta}$ represents the weights and biases of the network.

### 2.2 Bayesian Epistemic Uncertainty via Monte Carlo Dropout (MC-DO)
Standard neural networks yield single point predictions and are prone to overfitting small observational datasets. Following Gal & Ghahramani (2016), dropout with rate $p=0.3$ is retained **active during inference**:

$$\mathbf{W}^{(t)} = \mathbf{W} \odot \text{Bernoulli}(1-p), \quad t = 1, \dots, N_{\text{MC}}$$

By executing $N_{\text{MC}} = 100$ stochastic forward passes for a radial distance $r$, we obtain a sampling of predictions $\hat{V}^{(1)}(r), \dots, \hat{V}^{(N_{\text{MC}})}(r)$:

1. **Mean Rotation Curve**:
   $$\bar{V}(r) = \frac{1}{N_{\text{MC}}} \sum_{t=1}^{N_{\text{MC}}} \hat{V}^{(t)}(r)$$
2. **Epistemic (Model) Variance**:
   $$\sigma_{\text{MC}}^2(r) = \frac{1}{N_{\text{MC}}} \sum_{t=1}^{N_{\text{MC}}} \left( \hat{V}^{(t)}(r) - \bar{V}(r) \right)^2$$
3. **Predicted Statistical Uncertainty**:
   $$\bar{\sigma}_{\text{stat}}(r) = \frac{1}{N_{\text{MC}}} \sum_{t=1}^{N_{\text{MC}}} \hat{\sigma}_V^{(t)}(r)$$
4. **Total Uncertainty (Quadratic Combination)**:
   $$\sigma_{\text{total}}(r) = \sqrt{\sigma_{\text{MC}}^2(r) + \bar{\sigma}_{\text{stat}}^2(r)}$$

### 2.3 Genetic Algorithm (GA) Hyperparameter Optimization
To find the optimal network capacity for each galaxy, a Genetic Algorithm (`nnogada` library built on DEAP) searches the hyperparameter space:
* Number of hidden layers $\in [3, 4]$
* Neurons per layer $\in [50, 100, 150, 200]$
* Batch size $\in [4, 8, 16, 32]$

Candidate architectures are evaluated using the validation **Mean Squared Error (MSE)**:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^n \left( y_i - \hat{y}_i \right)^2$$

---

## 3. Computational & Coding Architecture

### 3.1 Neural Network Implementation (PyTorch)
```python
import torch
import torch.nn as nn

class RotationCurveANN(nn.Module):
    def __init__(self, num_hidden_layers=4, num_neurons=200, dropout_rate=0.3):
        super(RotationCurveANN, self).__init__()
        layers = []
        layers.append(nn.Linear(1, num_neurons))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_rate))
        
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(num_neurons, num_neurons))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            
        layers.append(nn.Linear(num_neurons, 2))  # Coupled outputs: [V(r), error(r)]
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

    def enable_mc_dropout(self):
        """Keep dropout active during inference."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
```

### 3.2 Feature Normalization & Training Protocol
- **Z-score Normalization**: Inputs $r$ and target velocities $V(r)$ are normalized: $r_{\text{norm}} = (r - \mu_r)/\sigma_r$.
- **Optimizer**: Adam with fixed learning rate $\eta = 5 \times 10^{-4}$.
- **Training Epochs**: $1000$ epochs per galaxy.

---

## 4. Benchmark Results & Group Classification

The paper classifies 17 THINGS galaxies into three distinct physical regimes:

| Group | Galaxies | Criterion | Physical Insight |
| :---: | :--- | :--- | :--- |
| **Group A** | DDO154, NGC2903, NGC3621, NGC2403 | Parametric NFW fits outperform ANN | Smooth, regular dynamics where theoretical NFW assumptions hold exceptionally well. |
| **Group B** | NGC3031, NGC2841, NGC3198, NGC5055 | ANN outperforms $\Upsilon_\star^{\text{fix}}$, but $\Upsilon_\star^{\text{free}}$ is best | Complex two-disk or bulge-dominated structures where stellar mass-to-light flexibility is required. |
| **Group C** | NGC4736, NGC925, NGC6946, NGC7793, NGC2976, NGC7331, NGC3521, NGC2366, IC2574 | ANN outperforms BOTH parametric NFW models | Highly irregular rotation curves, non-circular motions, or localized clumpiness where smooth parametric models fail. |

### Summary of Extracted Table 2 MSE Metrics
| Galaxy | Group | NFW Fix MSE | NFW Free MSE | ANN MSE | Winner |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DDO154** | A | **2.83** | 2.91 | 3.50 | NFW Fix |
| **NGC3621** | A | 12.24 | **10.17** | 22.10 | NFW Free |
| **NGC5055** | B | 239.95 | **42.84** | 124.15 | NFW Free |
| **NGC3521** | C | 335.01 | 225.37 | **62.68** | **ANN** |
| **NGC2976** | C | 25.88 | 11.15 | **3.10** | **ANN** |
| **IC2574** | C | 14.59 | 8.55 | **2.47** | **ANN** |
