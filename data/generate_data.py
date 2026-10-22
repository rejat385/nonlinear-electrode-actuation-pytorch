
import numpy as np
from pathlib import Path

NUM_SAMPLES = 10_000
NUM_ELECTRODES = 7
MAX_VOLTAGE = 100.0

rng = np.random.default_rng(42)

# Index 0: central electrode
# Indices 1-6: surrounding electrodes
adjacency = np.zeros((7, 7), dtype=np.float32)

# Every outer electrode neighbors the center.
for i in range(1, 7):
    adjacency[0, i] = 1.0
    adjacency[i, 0] = 1.0

# Adjacent outer electrodes also interact.
for i in range(1, 7):
    next_i = 1 + (i % 6)
    adjacency[i, next_i] = 1.0
    adjacency[next_i, i] = 1.0

# Normalize each row to average neighboring effects.
coupling = adjacency / adjacency.sum(axis=1, keepdims=True)

# Generate random voltage configurations.
'''
voltages = rng.uniform(
    low=0.0,
    high=MAX_VOLTAGE,
    size=(NUM_SAMPLES, NUM_ELECTRODES)
).astype(np.float32)

'''

# Generate 10,000 configurations with better coverage

n_random = 7000
n_single = 2000
n_pair = 1000

# 1. Random configurations
random_v = rng.uniform(
    0, MAX_VOLTAGE,
    size=(n_random, NUM_ELECTRODES)
).astype(np.float32)

# 2. Single-electrode configurations
single_v = np.zeros(
    (n_single, NUM_ELECTRODES),
    dtype=np.float32
)

for i in range(n_single):
    electrode = rng.integers(0, NUM_ELECTRODES)
    single_v[i, electrode] = rng.uniform(
        0, MAX_VOLTAGE
    )

# 3. Two-electrode configurations
pair_v = np.zeros(
    (n_pair, NUM_ELECTRODES),
    dtype=np.float32
)

for i in range(n_pair):
    electrodes = rng.choice(
        NUM_ELECTRODES,
        size=2,
        replace=False
    )

    pair_v[i, electrodes] = rng.uniform(
        0, MAX_VOLTAGE,
        size=2
    )

# Combine configurations
voltages = np.vstack([
    random_v, single_v, pair_v
])

# Shuffle the dataset
rng.shuffle(voltages)

v = voltages / MAX_VOLTAGE

# Nonlinear response of each electrode.
self_response = 0.7*v + 0.3*v**2 + 0.1*v**3

# Crosstalk from neighboring electrodes.
crosstalk = 0.12 * (v**2 @ coupling.T)

# Add small simulated measurement noise.
noise = rng.normal(
    0.0, 0.005, size=self_response.shape
).astype(np.float32)

deformation = self_response + crosstalk + noise

# Save data.
Path("data").mkdir(exist_ok=True)
np.savez(
    "data/electrode_dataset.npz",
    voltages=voltages,
    deformation=deformation
)

print("Dataset generated successfully!")
print("Voltages shape:", voltages.shape)
print("Deformation shape:", deformation.shape)
print("Example voltage input:", voltages[0].round(2))
print("Example deformation:", deformation[0].round(3))

data = np.load("data/electrode_dataset.npz")
for key, value in data.items():
    np.savetxt("data/electrode_dataset_" + key + ".csv", value)