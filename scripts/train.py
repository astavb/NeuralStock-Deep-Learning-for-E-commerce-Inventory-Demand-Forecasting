import random
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import LSTMModel, MLPModel

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

writer = SummaryWriter("runs/neural_stock")

df = pd.read_csv("data/ecommerce_inventory_demand.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date")

df = df.ffill()

df["units_sold"] = df["units_sold"].astype(int)

Q1 = df["unit_price"].quantile(0.25)
Q3 = df["unit_price"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

median_price = df["unit_price"].median()

df.loc[
    (df["unit_price"] < lower_bound)
    | (df["unit_price"] > upper_bound),
    "unit_price"
] = median_price

df["lag_7"] = df["units_sold"].shift(7)
df["lag_14"] = df["units_sold"].shift(14)

df["rolling_mean_7"] = (
    df["units_sold"].rolling(window=7).mean()
)

df["rolling_std_7"] = (
    df["units_sold"].rolling(window=7).std()
)

df["rolling_mean_30"] = (
    df["units_sold"].rolling(window=30).mean()
)

df["rolling_std_30"] = (
    df["units_sold"].rolling(window=30).std()
)

df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter

df["is_weekend"] = (
    df["date"].dt.dayofweek.isin([5, 6]).astype(int)
)

df = df.dropna()

df = pd.get_dummies(
    df,
    columns=["product_category", "day_of_week"],
    drop_first=True
)

X = df.drop(
    columns=[
        "units_sold",
        "date",
        "product_id"
    ]
)

y = df["units_sold"]

train_end = int(len(df) * 0.7)
val_end = int(len(df) * 0.85)

X_train = X.iloc[:train_end]
X_val = X.iloc[train_end:val_end]
X_test = X.iloc[val_end:]

y_train = y.iloc[:train_end]
y_val = y.iloc[train_end:val_end]
y_test = y.iloc[val_end:]


scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

def create_sequences(features, target, sequence_length):

    X_seq = []
    y_seq = []

    for i in range(len(features) - sequence_length):

        X_seq.append(features[i:i + sequence_length])

        y_seq.append(target.iloc[i + sequence_length])

    return np.array(X_seq), np.array(y_seq)

sequence_length = 21

X_train_seq, y_train_seq = create_sequences(
    X_train_scaled,
    y_train,
    sequence_length
)

X_val_seq, y_val_seq = create_sequences(
    X_val_scaled,
    y_val,
    sequence_length
)

X_test_seq, y_test_seq = create_sequences(
    X_test_scaled,
    y_test,
    sequence_length
)

X_train_tensor = torch.tensor(
    X_train_seq,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train_seq,
    dtype=torch.float32
)

X_val_tensor = torch.tensor(
    X_val_seq,
    dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val_seq,
    dtype=torch.float32
)

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=False
)

input_size = X_train_seq.shape[2]

lstm_model = LSTMModel(input_size)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    lstm_model.parameters(),
    lr=0.001
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

epochs = 100

best_val_loss = float("inf")

for epoch in range(epochs):

    lstm_model.train()

    epoch_loss = 0

    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()

        outputs = lstm_model(X_batch).squeeze()

        loss = criterion(outputs, y_batch)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    average_loss = epoch_loss / len(train_loader)

    lstm_model.eval()

    with torch.no_grad():

        val_outputs = lstm_model(
            X_val_tensor
        ).squeeze()

        val_loss = criterion(
            val_outputs,
            y_val_tensor
        )

    scheduler.step(val_loss)

    writer.add_scalar(
        "Training Loss",
        average_loss,
        epoch
    )

    writer.add_scalar(
        "Validation Loss",
        val_loss.item(),
        epoch
    )

    print(
        f"Epoch {epoch+1}/{epochs}, "
        f"Train Loss: {average_loss:.4f}, "
        f"Val Loss: {val_loss.item():.4f}"
    )

    if val_loss.item() < best_val_loss:

        best_val_loss = val_loss.item()

        torch.save(
            lstm_model.state_dict(),
            "models/lstm_model.pt"
        )

writer.close()

print("Model Training Completed")