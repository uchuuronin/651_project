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

Install llama.cpp separately for inference (refer to https://github.com/ggml-org/llama.cpp#installation)


## Usage
```bash
# Download datasets
python fetch_data.py --dataset triviaqa
python fetch_data.py --dataset popqa

# Start llama-server in a separate terminal before running inference:
llama-server -hf Qwen/Qwen2.5-1.5B-Instruct-GGUF --host 127.0.0.1 --port 4020

# Run inference
python run_inference.py # 50 examples, triviaqa, model=1.5b
python run_inference.py --lim 1000 # 1000 examples
python run_inference.py --lim 1000 --dataset popqa # popqa
python run_inference.py --lim 500 --model 7b # tag output as 7b (swap model in llama-server first)

# Run sweep 
python sweep.py # 50 examples, triviaqa, model=1.5b
python sweep.py --lim 1000 # full sweep
python sweep.py --lim 500 --model 7b # sweep 7b results

# Generate plots
python gen_plots.py # theoretical curves + reliability (50 ex)
python gen_plots.py --lim 1000 --dataset triviaqa # reliability on full dataset
python gen_empirical_plots.py # empirical vs theoretical (pilot + full)
python gen_empirical_plots.py --full-lim 500 --model 7b # compare 7b results

# Run tests
python run_tests.py
```