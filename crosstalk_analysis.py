
import numpy as np
import torch
import matplotlib.pyplot as plt

from models.mlp import ElectrodeMLP

# Load trained MLP
model = ElectrodeMLP()
model.load_state_dict(
    torch.load(
        "checkpoints/electrode_mlp_best.pth",
        map_location="cpu",
        weights_only=True
    )
)
model.eval()

# Recreate the synthetic coupling matrix
adjacency = np.zeros((7, 7), dtype=np.float32)

for i in range(1, 7):
    adjacency[0, i] = 1
    adjacency[i, 0] = 1

for i in range(1, 7):
    j = 1 + (i % 6)
    adjacency[i, j] = 1
    adjacency[j, i] = 1

coupling = adjacency / adjacency.sum(axis=1, keepdims=True)


def actual_response(v):
    """Synthetic physical response without noise."""
    own = 0.7*v + 0.3*v**2 + 0.1*v**3
    neighbor = 0.12 * (v**2 @ coupling.T)
    return own + neighbor


def predict_response(v):
    tensor = torch.tensor(
        v, dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        return model(tensor).numpy()[0]


# Test low and high central-electrode voltage
voltage_levels = [0.2, 0.9]

for level in voltage_levels:
    voltages = np.zeros(7, dtype=np.float32)
    voltages[0] = level

    actual = actual_response(voltages)
    predicted = predict_response(voltages)

    print(f"\nCentral voltage: {level*100:.0f}")
    print("Actual:   ", np.round(actual, 4))
    print("Predicted:", np.round(predicted, 4))

    x = np.arange(7)
    width = 0.35

    plt.figure(figsize=(9, 4))
    plt.bar(
        x - width/2, actual,
        width, label="Simulated physics"
    )
    plt.bar(
        x + width/2, predicted,
        width, label="PyTorch MLP"
    )

    plt.xticks(x, [f"E{i+1}" for i in range(7)])
    plt.xlabel("Electrode region")
    plt.ylabel("Deformation")
    plt.title(
        f"Central Electrode at {level*100:.0f}% Voltage"
    )
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
