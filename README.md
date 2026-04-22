

**ECE 57000 Track 1: TinyReproductions** | Purdue University | April 2026

---

## Project Overview

This project empirically validates that **A2C is a special case of PPO** (Huang et al., 2022) through progressive ablation. We train 5 model variants (A2C, PPO-0, PPO-1, PPO-2, PPO-3) across 3 seeds on CartPole-v1, measuring performance and structural similarity.

**Key Finding**: PPO-1 (K=1) achieves 3.5× closer parameter alignment with A2C (0.51 vs 1.81), but still outperforms A2C by 33% (471 vs 353 reward). Removing clipping causes catastrophic 60% performance drop.

---

## Quick Start

**Google Colab **:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YZXLp3gLH8Y_h5rTU8iqmgrMWQnwZ-2B)

1. Click badge above → Runtime → Run all
2. Wait ~30 minutes (CPU)
3. Results appear in notebook and save to `results/`

**Local Execution**:
```bash
pip install stable-baselines3[extra] gymnasium torch numpy matplotlib
python AI_Project_Checkpoint_2.py  # Or run .ipynb in Jupyter
```

---

## Dependencies

Automatically installed in Cell 1:
```python
!pip install stable-baselines3[extra] gymnasium --quiet
```

**Packages**: `stable-baselines3` (≥2.0), `gymnasium` (≥0.28), `torch`, `numpy`, `matplotlib`

---

## Code Structure

### Notebook Organization

```
AI_Project_Checkpoint_2.ipynb
│
├── Cell 1: Install dependencies (auto-install)
│
├── Cell 3-4: Imports & configuration (lines 9-26)
│
├── Cell 5: Helper functions (lines 27-85)
│   ├── set_global_seed() 
│   ├── make_env() 
│   ├── SimpleLearningTracker
│   └── train_with_tracking() 
│
├── Cell 6: Model configurations (lines 86-145)
│   ├── create_models() - AUTHOR WRITTEN
│   ├── A2C configuration (lines 91-97)
│   ├── PPO-0 baseline (lines 99-105)
│   ├── PPO-1: K=1 (lines 107-113)
│   ├── PPO-2: disable GAE (lines 115-122)
│   └── PPO-3: no clipping (lines 124-132)
│
├── Cell 7: Experiment loop (lines 146-180)
│   └── run_experiment() 
│
├── Cell 8: Visualization (lines 181-330)
│   ├── plot_learning_curves() 
│   ├── plot_final_performance() 
│   ├── plot_parameter_distances() 
│   └── save_results_json() 
│
└── Cell 9: Main execution (lines 331-415)
   
```

---

## Code Attribution

### **Written by me** (Lines 27-415)
All experimental code written for this project:
- Helper functions: `set_global_seed()`, `make_env()`, `SimpleLearningTracker`, `train_with_tracking()`
- Model configurations: `create_models()` with 5 ablation variants
- Experiment pipeline: `run_experiment()`, multi-seed training loop
- Visualization: All plotting functions (`plot_learning_curves()`, `plot_parameter_distances()`, etc.)
- Main execution: Complete experimental orchestration

### **Adapted from Literature**
Configuration parameters based on published papers (NO library code modified):
- **Line 112**: `n_epochs=1` - Adapted from Huang et al. (2022) for theoretical equivalence
- **Lines 120-121**: `gae_lambda=1.0`, `normalize_advantage=False` - Adapted from Huang et al. (2022)
- **Line 130**: `clip_range=0.0` - Adapted from Huang et al. (2022)
- **Line 100**: Default PPO params from Schulman et al. (2017)

### **Library API Calls** (NO modifications)
- **Lines 9-18**: Standard imports from documentation
- **Lines 91-132**: Stable-Baselines3 `A2C()` and `PPO()` instantiation - uses public API only
- **Line 76**: `model.learn()` - Stable-Baselines3 API call
- **Lines 79-81**: `evaluate_policy()` - Stable-Baselines3 utility function

**Important**: No modifications to Stable-Baselines3, Gymnasium, or PyTorch source code. Only configuration-level changes through public APIs.

---

## Dataset & Models

**Dataset**: CartPole-v1 (Gymnasium) - automatically downloaded with package  
**Pre-trained models**: None - all trained from scratch  
**Architecture**: 2-layer MLP, 64 units/layer, ReLU activation  
**Seeds**: [0, 42, 123] for reproducibility

---

## Expected Output

**Console**: Progress logs showing training for each model/seed  
**Files** (saved to `results/`):
- `learning_curves.png` - Training dynamics over 50k steps
- `parameter_distances.png` - Structural similarity to A2C  
- `final_performance.png` - Bar chart of final rewards
- `results.json` - Complete numerical results

**Expected Performance** (±5 tolerance):
| Model | Mean Reward | Std Dev |
|-------|-------------|---------|
| A2C | 352.83 | ±144.66 |
| PPO-0 | 500.00 | ±0.00 |
| PPO-1 | 470.92 | ±32.54 |
| PPO-2 | 494.43 | ±3.36 |
| PPO-3 | 200.62 | ±131.97 |

---

## Reproducibility

**Seed control** set at 3 levels:
1. Global: `np.random.seed()`, `torch.manual_seed()` (lines 28-30)
2. Environment: `env.reset(seed=)`, `env.action_space.seed()` (lines 36-39)
3. Models: `A2C(..., seed=)`, `PPO(..., seed=)` (lines 91-132)

**Runtime**: ~70 min (Colab CPU), ~60 min (local CPU)  
**Platform**: Cross-platform (Linux/macOS/Windows)

---

## References

1. Schulman et al., "Proximal Policy Optimization", arXiv:1707.06347, 2017
2. Mnih et al., "Asynchronous Methods for Deep RL", ICML 2016
3. Huang et al., "A2C is a special case of PPO", arXiv:2205.09123, 2022
4. Raffin et al., "Stable-Baselines3", JMLR 2021

---

## License

Project: Academic use (ECE 57000) | Dependencies: MIT (SB3, Gymnasium), BSD (PyTorch)

---

