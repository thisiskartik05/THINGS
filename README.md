# Data-Driven Modeling of Galactic Rotation Curves with Artificial Neural Networks

[![arXiv](https://img.shields.io/badge/arXiv-2404.05833-b31b1b.svg)](https://arxiv.org/abs/2404.05833)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **Full Reproduction and Independent Extension of arXiv:2404.05833v2**  
> *Gabriela Garcia-Arroyo, Isidro Gómez-Vargas, J. Alberto Vázquez (2024)*  

---

## 📌 Project Overview

This repository contains a full, independent reproduction of the non-parametric Artificial Neural Network (ANN) pipeline with **Monte Carlo Dropout (MC-DO)** and **Genetic Algorithm (GA)** hyperparameter optimization for modeling the rotation curves of **17 spiral galaxies** from *The HI Nearby Galaxy Survey (THINGS)*.

### Key Scientific Takeaways
1. **Data-Driven Modeling**: ANNs fit global, continuous rotational velocity mappings $r \to [V(r), \sigma_V(r)]$ directly from observational data without predefined theoretical mass profiles.
2. **Epistemic Uncertainty Quantification**: Monte Carlo Dropout retains active dropout layers during inference ($N_{\text{MC}} = 100$ stochastic forward passes), providing robust Bayesian uncertainty estimates.
3. **Physical Classification**: Galaxies are classified into three distinct physical regimes:
   - **Group A** (e.g., **DDO154**, NGC3621): Simple, regular dynamics where theoretical NFW parametric models outperform ANNs.
   - **Group B** (e.g., NGC5055): Complex disk/bulge structures where flexible mass-to-light ratios ($\Upsilon_\star^{\text{free}}$) are required.
   - **Group C** (e.g., **NGC3521**, NGC2976, IC2574): Irregular rotation curves, non-circular motions, or localized clumpiness where ANNs significantly outperform parametric NFW models.

---

## 🗂 Repository Structure

```
THINGS/
├── README.md                           # Master repository documentation
├── reproduce_rotation_curves.ipynb     # Self-contained reproduction notebook
├── reproduction_plan.md                # 5-phase reproduction roadmap
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Git ignore configuration
├── data/                               # Observational rotation curves & CSV benchmarks
│   ├── Curves/                         # Raw .curve.02 data files (17 THINGS galaxies)
│   ├── MassModels/                     # Parametric mass model files (.dat)
│   ├── table1_architectures.csv        # Extracted ANN hyperparameter architectures
│   ├── table2_mse.csv                  # Benchmark MSE values (NFW fix, NFW free, ANN)
│   └── table3_groups.csv               # Group A, B, C classifications
└── docs/                               # Comprehensive theoretical handouts
    ├── master_handout.md               # Markdown master handout with physics/math/code
    ├── master_handout.tex              # LaTeX source for master handout
    └── master_handout.pdf              # Compiled PDF master handout (via pdflatex)
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/thisiskartik05/THINGS.git
cd THINGS
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Reproduction Notebook
Open and run [`reproduce_rotation_curves.ipynb`](file:///Users/kartikkhare/Desktop/THINGS/reproduce_rotation_curves.ipynb):
```bash
jupyter notebook reproduce_rotation_curves.ipynb
```

---

## 📜 Citation

If you use this codebase or theoretical handouts, please cite the original authors' paper:
```bibtex
@article{GarciaArroyo2024,
  title={Data-driven modeling of rotation curves with artificial neural networks},
  author={Garcia-Arroyo, Gabriela and G{\'o}mez-Vargas, Isidro and V{\'a}zquez, J. Alberto},
  journal={arXiv preprint arXiv:2404.05833},
  year={2024}
}
```
