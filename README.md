# EN4554 Assignment 2: CNNs on CIFAR-10

**Dissanayake D.M.A.D | 220134K**

A PyTorch notebook for EN4554 Assignment 2. It covers backpropagation, optimizers, regularization, residual networks and the design of a small CNN, all on CIFAR-10.

## Files

- `Assignment_2_220134K.ipynb`: the notebook with the code, results and explanations
- `220134K_a02.pdf`: the report exported from the notebook

## Tasks

1. **Backpropagation:** Linear, ReLU and softmax cross-entropy with manual forward and backward passes, gradient checks against autograd and finite differences, MLP training, and gradient norms per layer (sigmoid vs ReLU).
2. **Optimizers:** custom SGD with momentum and Adam, trajectories on a narrow quadratic valley, and learning-rate sensitivity on a small CNN.
3. **Regularization and residual connections:** weight decay, dropout, flip + crop, cutout and mixup on 5k training images, and plain vs residual networks of depth 8, 20 and 56.
4. **Best CNN (optional):** a ResNet with 464k parameters trained for 20 epochs, with an ablation study.

## Main results

| Experiment | Result |
|---|---|
| MLP (Task 1c) | 51.2% best validation accuracy |
| Small CNN, best optimizer (Task 2c) | Adam with lr = 1e-3: 77.2% |
| 5k images, best regularizer (Task 3a) | flip + crop: 76.0% |
| ResNet-56 vs plain-56 (Task 3b) | 92.4% vs 75.7% validation accuracy |
| Final CNN (Task 4) | **89.9% test accuracy** |

## How to run

1. Open the notebook in Google Colab and select a T4 GPU (Runtime → Change runtime type).
2. Run all cells. CIFAR-10 is downloaded to Google Drive (`MyDrive/EN4554_A2`), and every training run is saved there, so later runs load the results instead of retraining.
3. For a fast check, set `QUICK_TEST = True` in the first code cell (1 epoch per experiment).

Requirements: `torch`, `torchvision`, `numpy`, `matplotlib` and `pandas`. All of them are preinstalled on Colab.
