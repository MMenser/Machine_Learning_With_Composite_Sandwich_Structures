# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# %%
INPUT_SIZE = 3 # thickness, height, angle
LEARNING_RATE = 0.001
BATCH_SIZE = 32 # 
NUM_EPOCHS = 1000

TEST_SIZE = 0.2
RANDOM_STATE = 42
device = "cpu"

# %%
df = pd.read_csv(r'../processed_bending_stiffness.csv')

# Remove duplicates
initial_count = len(df)
df = df.drop_duplicates()
removed_count = initial_count - len(df)
if removed_count > 0:
    print(f"Removed {removed_count} duplicate row(s) from the dataset.")
print(f"Shape of dataset after removing duplicates: {df.shape}")

X = df[['Thickness', 'Height', 'Angle (deg)']]
y = df['Bending_Stiffness']

# 1. First split: Separate 10% for the final Test set
# This leaves 90% in X_temp
X_temp, X_test, y_temp, y_test = train_test_split(
    X.values, 
    y.values, 
    test_size=0.10,  # 10% for test
    random_state=RANDOM_STATE
)

# 2. Second split: Split the remaining 90% into Train (70%) and Val (20%)
# To get 20% of the TOTAL, we take 2/9ths of the 90% (approx 0.2222)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.2222, 
    random_state=RANDOM_STATE
)

print(f"\nDataset Split:")
print(f"Train set size: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
print(f"Validation set size: {len(X_val)} ({len(X_val)/len(df)*100:.1f}%)")
print(f"Test set size: {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")

DATASET_SIZE = len(df)

# %%
model = nn.Sequential(
        nn.Linear(INPUT_SIZE, 64),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(p=0.2),

        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        
        # Output Layer
        nn.Linear(16, 1)
).to(device)


# %%
# Convert TRAINING
X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Convert VALIDATION
X_val_tensor = torch.from_numpy(X_val.astype(np.float32))
y_val_tensor = torch.from_numpy(y_val.astype(np.float32)).unsqueeze(1)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Convert TESTING
X_test_tensor = torch.from_numpy(X_test.astype(np.float32))
y_test_tensor = torch.from_numpy(y_test.astype(np.float32)).unsqueeze(1)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %%
mae_criterion = nn.L1Loss()  # Mean Absolute Error Loss
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

trainLoss = []
validationLoss = []

print("\n--- Starting Training ---")
for epoch in range(NUM_EPOCHS):
    # ========== TRAINING PHASE ==========
    model.train()  # Set model to training mode
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * targets.size(0)
    
    avg_train_loss = running_loss / len(X_train)
    trainLoss.append(avg_train_loss)
    
    # ========== VALIDATION PHASE ==========
    model.eval()
    running_val_loss = 0.0
    val_samples = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            val_loss = criterion(outputs, targets)
            running_val_loss += val_loss.item() * targets.size(0)
            val_samples += targets.size(0)
    
    avg_val_loss = running_val_loss / val_samples
    validationLoss.append(avg_val_loss)
    
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

# Plot training curves
plt.figure(figsize=(10, 6))
plt.plot(trainLoss, label='Training Loss', color='blue')
plt.plot(validationLoss, label='Validation Loss', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()
# --- Final Evaluation on TEST SET ---
print("\n--- Final Evaluation on Test Set ---")
model.eval()
total_squared_error = 0.0
total_absolute_error = 0.0
total_samples = 0
all_targets = []
all_predictions = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs)
        
        # MSE loss
        mse_loss = criterion(outputs, targets)
        total_squared_error += mse_loss.item() * targets.size(0)
        
        # MAE loss
        mae_loss = mae_criterion(outputs, targets)
        total_absolute_error += mae_loss.item() * targets.size(0)
        
        total_samples += targets.size(0)
        all_targets.append(targets.cpu())
        all_predictions.append(outputs.cpu())

# Concatenate all targets and predictions
targets_tensor = torch.cat(all_targets)
predictions_tensor = torch.cat(all_predictions)

# Calculate metrics
mean_squared_error = total_squared_error / total_samples
root_mean_squared_error = np.sqrt(mean_squared_error)
mean_absolute_error = total_absolute_error / total_samples

# R-squared
ss_residual = total_squared_error
target_mean = targets_tensor.mean()
ss_total = ((targets_tensor - target_mean) ** 2).sum().item()
r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0.0

print(f'\nTest Set Metrics:')
print(f'MSE: {mean_squared_error:.4f}')
print(f'RMSE: {root_mean_squared_error:.4f}')
print(f'MAE: {mean_absolute_error:.4f}')
print(f'R-squared (R²): {r_squared:.4f}')
print("\nTraining and Evaluation complete.")

# %%
torch.save(model.state_dict(), 'model_weight.pth')

# %%
# Training set predictions
array_y_train_pred = model(X_train_tensor).detach().numpy()
array_y_train = y_train_tensor.detach().numpy()

# Validation set predictions
array_y_val_pred = model(X_val_tensor).detach().numpy()
array_y_val = y_val_tensor.detach().numpy()

# Test set predictions
array_y_test_pred = model(X_test_tensor).detach().numpy()
array_y_test = y_test_tensor.detach().numpy()

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(array_y_train, array_y_train_pred, 'o', alpha=0.6)
plt.plot([array_y_train.min(), array_y_train.max()], 
            [array_y_train.min(), array_y_train.max()], 
            'r--', linewidth=2)
plt.xlabel('Actual Bending Stiffness')
plt.ylabel('Predicted Bending Stiffness')
plt.title('Training Set: Actual vs Predicted')
plt.grid("on")

plt.subplot(1, 3, 2)
plt.plot(array_y_val, array_y_val_pred, 'o', alpha=0.6, color='orange')
plt.plot([array_y_val.min(), array_y_val.max()], 
            [array_y_val.min(), array_y_val.max()], 
            'r--', linewidth=2)
plt.xlabel('Actual Bending Stiffness')
plt.ylabel('Predicted Bending Stiffness')
plt.title('Validation Set: Actual vs Predicted')
plt.grid("on")

plt.subplot(1, 3, 3)
plt.plot(array_y_test, array_y_test_pred, 'o', alpha=0.6, color='green')
plt.plot([array_y_test.min(), array_y_test.max()], 
            [array_y_test.min(), array_y_test.max()], 
            'r--', linewidth=2)
plt.xlabel('Actual Bending Stiffness')
plt.ylabel('Predicted Bending Stiffness')
plt.title('Testing Set: Actual vs Predicted')
plt.grid("on")

plt.tight_layout()
plt.show()

# %%
print(model(torch.tensor([[4.91, 33.80, 47.35]], dtype=torch.float32)))  # Example bending stiffness input