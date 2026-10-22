import time
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

from models.mlp import ElectrodeMLP

# 1. Load generated data
data = np.load("data/electrode_dataset.npz")

X = (data["voltages"]/100).astype(np.float32)
y = data["deformation"].astype(np.float32)

# 2. Split into train, validation and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

def evaluate(name, predictions, elapsed):
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    print(
        f"{name:20s} | "
        f"MSE: {mse:.7f} | "
        f"RMSE: {rmse:.7f} | "
        f"Prediction: {elapsed*1000:.3f} ms"
    )

# Model 1 : Linear Regression
linear = LinearRegression()
linear.fit(X_train, y_train)

start = time.perf_counter()
linear_pred = linear.predict(X_test)
linear_time = time.perf_counter() - start

evaluate("Linear Regression", linear_pred, linear_time)

# Model 2 : kNN
knn = KNeighborsRegressor(n_neighbors=5, weights="distance")
knn.fit(X_train, y_train)

start = time.perf_counter()
knn_pred = knn.predict(X_test)
knn_time = time.perf_counter() - start

evaluate("kNN Regression", knn_pred, knn_time)

# Model 3 : Pytorch MLP
model = ElectrodeMLP()
model.load_state_dict(torch.load("checkpoints\electrode_mlp_best.pth", map_location="cpu", weights_only=True))

model.eval()
X_tensor = torch.from_numpy(X_test)

with torch.no_grad():
    # Warm up PyTorch before timing
    _ = model(X_tensor)

    start = time.perf_counter()
    mlp_pred = model(X_tensor).numpy()
    mlp_time = time.perf_counter() - start

evaluate("PyTorch MLP", mlp_pred, mlp_time)
