# CMEC Sandwich Panel

Machine learning experiments for sandwich panel bending stiffness. The repo includes raw bending-stiffness CSV data, preprocessing/augmentation work, and PyTorch models for predicting stiffness or design parameters.

## Project Structure

- `Data/Raw/bending_stiffness/` - raw bending stiffness CSV files.
- `Models/MLP/processed_bending_stiffness.csv` - processed dataset used by the MLP models.
- `Models/MLP/Parameters_To_Stiffness/mlp.py` - predicts bending stiffness from thickness, height, and angle.
- `Models/MLP/Stiffness_To_Parameters/mlp.py` - predicts thickness, height, and angle from bending stiffness.
- `Models/VAE/cvae.py` - conditional VAE experiment using the processed stiffness data.
- `Models/Augmentation/` - image/data preprocessing and augmentation notebooks/scripts.

## Setup

Use Python 3 and install the main dependencies:

```bash
pip install torch numpy pandas scikit-learn matplotlib pillow
```

## Running Models

Run scripts from their own folders so relative dataset paths resolve correctly:

```bash
cd Models/MLP/Parameters_To_Stiffness
python mlp.py
```

```bash
cd Models/VAE
python cvae.py
```

Some scripts contain absolute local paths and may need path updates before running on another machine.
