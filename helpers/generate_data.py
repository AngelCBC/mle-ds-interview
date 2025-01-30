# %%
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

x_offset = 0.25
y_offset = -0.75

N = 12000  # Number of rows
df = pd.DataFrame(
    {
        "theta": np.random.uniform(0, 2 * np.pi, N),
        "r": np.sqrt(np.random.uniform(0, 1, N)),
        "noise": np.random.normal(0, 1.1, N),
    }
)


# %%
def sigmoid(x):
    k = 0.15
    s = 1 / (1 + np.exp(-(2 * x - 1) / k))
    return s


# un-normalised data
x = np.linspace(-0.1, +1.1, 100)

plt.plot(x, sigmoid(x))
plt.show()


# %%
def assign_label(r):
    s = sigmoid(r)
    return 0 if random.uniform(0, 1) > s else 1


df["label"] = df["r"].apply(assign_label)
df["label"].value_counts(normalize=True)

df["x"] = df["r"] * np.cos(df["theta"]) + x_offset
df["y"] = 1 / 2 * df["r"] * np.sin(df["theta"]) + y_offset

# %%
plt.figure(figsize=(8, 6))
ax = plt.subplot(111, projection="polar")
scatter = ax.scatter(df["theta"], df["r"], c=df["label"], cmap="viridis", alpha=0.75, s=10)
plt.colorbar(scatter, label="Label")
ax.set_title("Polar Scatter Plot by Label")
plt.show()

# %%
plt.figure(figsize=(8, 6))
ax = plt.subplot(111)
scatter = ax.scatter(df["x"], df["y"], c=df["label"], cmap="viridis", alpha=0.75, s=10)
plt.colorbar(scatter, label="Label")
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_title("Cartesian Scatter Plot by Label")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
plt.show()

# %%
df_train = df.iloc[: int((5 / 6) * N)].copy()
df_test = df.iloc[int((5 / 6) * N) :].copy()
print("Shape of new dataframes - {} , {}".format(df_train.shape, df_test.shape))

# Replace 0.1% of df_train values with NaN
total_values = df_train.size
num_nulls = int(total_values * 0.001)

# Randomly choose indices to replace with NaN
null_indices_x = np.random.randint(0, df_train.shape[0], num_nulls)
null_indices_y = np.random.randint(0, df_train.shape[0], num_nulls)

df_train.loc[null_indices_x, "x"] = np.nan
df_train.loc[null_indices_y, "y"] = np.nan
print(df_train.isna().sum())
# %%
select_cols = ["x", "y", "noise", "label"]
rename_mappings = {"x": "V0", "y": "V1", "noise": "V2"}
df_train[select_cols].rename(columns=rename_mappings).to_csv("train_data.csv", index=False)
df_test[select_cols].rename(columns=rename_mappings).to_csv("test_data.csv", index=False)
