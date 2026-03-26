# Optimal Loss-Aware Abstention Thresholds for LLMs
This project is part of required coursework for COMPSCI 651 at UMass Amherst, taken at Spring 2026  

We derive closed-form optimal abstention thresholds for LLMs under three loss functions and test whether they predict the empirically optimal threshold on real models. Current methods treat the threshold as a grid-searched hyperparameter with no theoretical basis. We show it can be computed directly from the abstention cost.

We are considering the following thresholds for our emperical analysis

| Loss | Formula | Domain |
|---|---|---|
| Asymmetric utility | τ_U(λ) = λ/(1+λ) | λ > 0 |
| Brier score | τ_B(λ) = (1+√(1−4λ))/2 | λ ∈ [0, 0.25] |
| Cross-entropy | τ_CE(λ) = H⁻¹(λ) | λ ∈ [0, ln2] |

Ordering: τ_CE > τ_B > τ_U across the valid domain.


## Setup
```bash
conda create -n cs651 python=3.10 -y
conda activate cs651
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Running
```bash
python pull_data.py # downloads dataset csv locally
python run_inference.py # runs interences
```