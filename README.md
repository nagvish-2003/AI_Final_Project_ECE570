
# Progressive Ablation Study: PPO to A2C Convergence

**ECE 57000 Course Project - Track 1: TinyReproductions**

This project empirically validates and extends the theoretical result that Advantage Actor-Critic (A2C) is a special case of Proximal Policy Optimization (PPO) when configured with a single update epoch.

## Project Overview

We conduct a systematic ablation study demonstrating how PPO's learned parameters and behavior converge toward A2C as key mechanisms are progressively removed:

- **PPO-0**: Default PPO (baseline)
- **PPO-1**: Single epoch updates (K=1)
- **PPO-2**: Single epoch + disable GAE + no advantage normalization
- **PPO-3**: Single epoch + disable GAE + no normalization + no clipping
- **A2C**: Standard A2C baseline

## Key Findings

1. **Structural Convergence**: PPO-1 achieves 3.5× closer parameter alignment with A2C compared to default PPO (0.51 vs 1.81 max parameter difference)

2. **Performance Divergence**: Despite structural similarity, PPO-1 outperforms A2C by 33% (471 vs 353 mean reward)

3. **Critical Role of Clipping**: Removing clipping (PPO-3) causes catastrophic failure with 60% performance drop

4. **Training Stability**: A2C exhibits high variance (±145) that PPO's mechanisms successfully mitigate

## Quick Start

### Run in Google Colab

Click here to open and run the complete experiment in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YZXLp3gLH8Y_h5rTU8iqmgrMWQnwZ-2B)

The notebook contains all code cells organized and ready to run. Simply execute cells in order.

### Local Installation

If running locally:

```bash
pip install stable-baselines3[extra] gymnasium torch numpy matplotlib
```

## Notebook Structure

The Google Colab notebook is organized into the following sections:

### **Cell 1: Install Dependencies**
```python
!pip install stable-baselines3[extra] gymnasium --quiet
```

### **Cell 2: Mount Google Drive**
Saves results permanently to your Google Drive.

### **Cell 3-8: Setup Code**
- Import libraries
- Configuration parameters
- Helper functions
- Model creation functions
- Training and evaluation functions
- Plotting functions

### **Cell 9: Run Experiments**
Main execution cell that:
- Trains all 5 model variants across 3 random seeds
- Generates learning curves and performance plots
- Computes parameter distances from A2C
- Saves results to `results/` directory

### **Cell 10-11: View Results** (Optional)
- Display generated plots
- Download results as ZIP

## Expected Runtime

- **Environment**: CartPole-v1
- **Total runtime**: ~60-75 minutes on CPU (Google Colab free tier)
- **Per model per seed**: ~3-4 minutes

## Output Files

The experiment generates the following files in the `results/` directory:

## Reproducibility

### Random Seeds

The project uses three fixed seeds (0, 42, 123) for reproducibility. Seeds are set for:
- NumPy random number generator
- PyTorch random number generator  
- Gymnasium environment initialization

### Expected Results

With the provided seeds, you should observe:

| Model | Mean Reward | Std Dev |
|-------|-------------|---------|
| A2C | ~353 | ±145 |
| PPO-0 | 500 | ±0 |
| PPO-1 | ~471 | ±33 |
| PPO-2 | ~494 | ±3 |
| PPO-3 | ~201 | ±132 |

**Note**: Minor variations (±5 reward points) may occur due to environment stochasticity and platform differences.

## Code Attribution

### Original Implementation

This project uses [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for PPO and A2C implementations:

- **Library**: Stable-Baselines3 v2.0+
- **License**: MIT License
- **Citation**: Raffin et al., "Stable-Baselines3: Reliable Reinforcement Learning Implementations", JMLR 2021

### Authored Code

All experimental code was written specifically for this project:

**Main Components**:
- `create_models()` - Progressive ablation configuration
- `train_with_tracking()` - Learning curve tracking
- `run_experiment()` - Multi-seed experiment loop
- `compute_parameter_distances()` - Parameter distance metric
- All plotting functions
- Main execution logic

### Adapted Code

Configuration parameters for PPO variants are adapted from:
- Huang et al., "A2C is a special case of PPO", arXiv 2022 (for theoretical equivalence settings)
- Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 2017 (for default PPO settings)

## Experimental Design

### Model Configurations

**A2C Baseline**:
```python
A2C("MlpPolicy", env, seed=seed, verbose=0, n_steps=128)
```

**PPO-0 (Default)**:
```python
PPO("MlpPolicy", env, seed=seed, verbose=0)
# Uses: n_epochs=10, gae_lambda=0.95, normalize_advantage=True, clip_range=0.2
```

**PPO-1 (Single Epoch)**:
```python
PPO("MlpPolicy", env, seed=seed, verbose=0, n_epochs=1)
```

**PPO-2 (Disable GAE & Normalization)**:
```python
PPO("MlpPolicy", env, seed=seed, verbose=0, 
    n_epochs=1, gae_lambda=1.0, normalize_advantage=False)
```

**PPO-3 (Remove Clipping)**:
```python
PPO("MlpPolicy", env, seed=seed, verbose=0,
    n_epochs=1, gae_lambda=1.0, normalize_advantage=False, clip_range=0.0)
```

### Evaluation Metrics

1. **Performance Metrics**:
   - Final episodic reward (mean ± std over 20 episodes)
   - Learning curves (evaluated every 2,000 steps)
   - Across-seed variance

2. **Structural Similarity Metric**:
   d(model, A2C) = max |θ_model - θ_A2C|
Maximum absolute parameter difference across all policy network parameters.

## Troubleshooting

### Common Issues

**Issue**: "RuntimeError: CUDA out of memory"
- **Solution**: CartPole runs fine on CPU. Colab defaults to CPU for this notebook.

**Issue**: Results differ slightly from paper
- **Solution**: Minor variations (±5 points) are normal due to environment stochasticity. Run with more seeds for tighter confidence intervals.

**Issue**: Training takes longer than expected
- **Solution**: Free Colab tier may slow down after extended use. Consider Colab Pro or run locally.

### Getting Help

For questions about:
- **Implementation**: Check the code comments in the notebook
- **Stable-Baselines3**: See [official documentation](https://stable-baselines3.readthedocs.io/)
- **Project requirements**: Refer to the course project guidelines

## Hardware Requirements

- **Minimum**: Single CPU core, 2GB RAM
- **Recommended**: 4+ CPU cores, 4GB RAM (faster execution)
- **GPU**: Not required (CartPole is CPU-efficient)

## References

1. Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347, 2017
2. Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning", ICML 2016
3. Huang et al., "A2C is a special case of PPO", arXiv:2205.09123, 2022
4. Raffin et al., "Stable-Baselines3", JMLR 2021

## Citation

If you use this code for academic work, please cite:

```bibtex
@misc{ppo_a2c_ablation2026,
  title={Progressive Ablation Study: Empirical Analysis of PPO to A2C Convergence},
  author={ECE 57000 Course Project},
  year={2026},
  howpublished={Purdue University}
}
```

## License

This code is provided for academic use in ECE 57000. The Stable-Baselines3 library is licensed under MIT License.

## Acknowledgments

- **Stable-Baselines3 Team**: For providing robust RL algorithm implementations
- **OpenAI Gymnasium**: For the CartPole-v1 environment
- **Huang et al.**: For the theoretical foundation ("A2C is a special case of PPO")
- **Course Staff**: For guidance and feedback throughout the project

## Project Structure
├── README.md                           # This file
├── AI_Project_Checkpoint_2.ipynb      # Google Colab notebook (main code)
├── results/                           # Generated output files
│   ├── learning_curves.png
│   ├── final_performance.png
│   ├── parameter_distances.png
│   └── results.json
└── paper/
├── main.tex                       # ICLR-style paper
├── references.bib                 # Bibliography
├── learning_curves.png            # Figure 1
└── parameter_distances.png        # Figure 2
