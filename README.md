# Adaptive Uncertainty Estimation via High-Dimensional Testing on Latent Representations

Code implementations for "Adaptive Uncertainty Estimation via High-Dimensional Testing on Latent Representations"

## Trainers
Trainers training our method and some of the baselines are provided in `./trainers`. Other baselines are included in `./baselines`. Users can customize their own trainers and import them in `main.py`.

## Architectures
Both the BNN and frequentist encoder architectures (e.g., LeNet) can be found in the `./models` directory. Users can also customize own encoder architectures.

## Get Started
Create a yaml config file in the `./configs` directory (examples can be found in the same directory), and run the following codes to run an experiment
```
python main.py
```
Results will be saved in `./checkpoints`.
