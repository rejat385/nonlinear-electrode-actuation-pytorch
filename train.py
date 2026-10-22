import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from models.mlp import ElectrodeMLP

torch.manual_seed(42)

# 1. Load generated data
data = np.load("data/electrode_dataset.npz")

X = (data["voltages"]/100).astype(np.float32)
y = data["deformation"].astype(np.float32)

# 2. Split into train, validation and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# 3. Convert to Pytorch tensors
def to_tensor(array):
    return torch.from_numpy(array)

train_dataset = TensorDataset(to_tensor(X_train), to_tensor(y_train))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

X_val = to_tensor(X_val)
X_test = to_tensor(X_test)
y_val = to_tensor(y_val)
y_test = to_tensor(y_test)

# 4. Initialize model, loss function, optimizer
model = ElectrodeMLP()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=10,
    min_lr=1e-5
)

# 5. Training loop
epochs = 150
train_losses = []
val_losses = []

best_val_loss = float("inf")
Path("checkpoints").mkdir(exist_ok=True)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for inputs, target in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += (loss.item() * inputs.size(0))

    train_loss = running_loss / len(train_dataset)

    # Validation
    model.eval()
    with torch.no_grad():
        eval_pred = model(X_val)
        val_loss = criterion(eval_pred, y_val).item()
        scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "checkpoints/electrode_mlp_best.pth"
        )

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if (epochs + 1) % 10 == 0:
        print(f"Epoch {epochs + 1:3d} | train_MSE: {train_loss:.6f} | val_MSE: {val_loss:.6f}")

# 6. Test
model.load_state_dict(
    torch.load(
        "checkpoints/electrode_mlp_best.pth",
        weights_only=True
    )
)

model.eval()
with torch.no_grad():
    test_pred = model(X_test)
    test_loss = criterion(test_pred, y_test).item()
    test_rmse = np.sqrt(test_loss)

print(f"\ntest_MSE: {test_loss:.6f} | test_RMSE: {test_rmse:.6f}")

# 7. Save
Path("checkpoints").mkdir(exist_ok=True)
torch.save(
    model.state_dict(),
    "checkpoints/electrode_mlp.pth"
)

# 8. Visualize learning
plt.plot(train_losses, label="Training loss")
plt.plot(val_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Electrode MLP Training")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.show()