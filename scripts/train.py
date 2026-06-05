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

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
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

sequence_length = 14

X_train_seq, y_train_seq = create_sequences(
    X_train_scaled,
    y_train,
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

epochs = 20

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

    writer.add_scalar(
        "Training Loss",
        average_loss,
        epoch
    )

    print(
        f"Epoch {epoch+1}/{epochs}, "
        f"Loss: {average_loss:.4f}"
    )

torch.save(
    lstm_model.state_dict(),
    "models/lstm_model.pt"
)

writer.close()

print("Model Training Completed")