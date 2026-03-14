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

## How to Run
1. Download `anime.csv` and `rating.csv` from [Kaggle](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)
2. Place them in the same folder as the notebook
3. Run all cells in `anime_autoencoder.ipynb`
```

---

### Step 4 — Add a `.gitignore` (optional but good)
Click **"Add file"** → **"Create new file"** → name it `.gitignore` and add:
```
*.csv
__pycache__/
.ipynb_checkpoints/
```
This prevents accidentally uploading the large CSV data files.

---

That's it! Your project will be live at:
```
https://github.com/YOUR_USERNAME/anime-recommender-autoencoder
