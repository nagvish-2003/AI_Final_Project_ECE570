# AI_Final_Project_ECE570
ECE570 Final Project

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

### Run in Google Colab (Recommended)

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

### **Cell 2: Mount Google Drive** (Optional)
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
