# Anime Recommendation System using Stacked Denoising Autoencoder

A collaborative filtering model built with PyTorch that uses a 
Stacked Denoising Autoencoder (SDAE) to predict anime ratings 
and power a recommendation system.

## Dataset
- **anime.csv** — anime metadata
- **rating.csv** — user ratings (filtered to 5,000 active users)
- Matrix sparsity: ~95%

## Model Architecture
- Input → Linear(128) → Tanh → Linear(20) → Tanh *(encoder)*
- Linear(128) → Tanh → Linear(input) → Sigmoid *(decoder)*
- Bottleneck: 20 dimensions
- Optimizer: RMSprop | Loss: Masked MSE

## Results
| Metric | Score |
|--------|-------|
| Test RMSE | 2.4467 |
| Test MAE | 2.1804 |

## Requirements
```
torch
pandas
numpy
scikit-learn
matplotlib
```

