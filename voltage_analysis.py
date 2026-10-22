
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

from models.mlp import ElectrodeMLP

# Load and normalize data
data = np.load("data/electrode_dataset.npz")
X = (data["voltages"] / 100.0).astype(np.float32)
y = data["deformation"].astype(np.float32)

# Same split as the previous experiment
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

# Train baseline models
linear = LinearRegression().fit(X_train, y_train)
knn = KNeighborsRegressor(
    n_neighbors=5, weights="distance"
).fit(X_train, y_train)

# Load trained neural network
mlp = ElectrodeMLP()
mlp.load_state_dict(
    torch.load(
        "checkpoints/electrode_mlp_best.pth",
        map_location="cpu",
        weights_only=True
    )
)
mlp.eval()

with torch.no_grad():
    mlp_predictions = mlp(
        torch.from_numpy(X_test)
    ).numpy()

predictions = {
    "Linear Regression": linear.predict(X_test),
    "kNN": knn.predict(X_test),
    "PyTorch MLP": mlp_predictions
}

# Divide by maximum voltage in each configuration.
# These are illustrative analysis categories.
max_voltage = X_test.max(axis=1)

regions = {
    "Low": max_voltage < 0.6,
    "Medium": (max_voltage >= 0.6) & (max_voltage < 0.85),
    "High": max_voltage >= 0.85
}

results = {}

for model_name, y_pred in predictions.items():
    scores = []

    for region_name, mask in regions.items():
        count = mask.sum()

        if count == 0:
            rmse = np.nan
        else:
            rmse = np.sqrt(
                mean_squared_error(
                    y_test[mask],
                    y_pred[mask]
                )
            )

        scores.append(rmse)

        print(
            f"{model_name:18s} | "
            f"{region_name:6s} | "
            f"N={count:4d} | RMSE={rmse:.6f}"
        )

    results[model_name] = scores

# Plot grouped RMSE comparison
labels = list(regions.keys())
x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))

for i, (name, scores) in enumerate(results.items()):
    ax.bar(
        x + (i - 1) * width,
        scores,
        width,
        label=name
    )

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("Voltage region")
ax.set_ylabel("RMSE")
ax.set_title("Model Error Across Voltage Regions")
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
