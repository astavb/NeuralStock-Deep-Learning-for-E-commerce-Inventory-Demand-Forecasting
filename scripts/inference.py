import joblib
import numpy as np
import pandas as pd

import torch

from model import LSTMModel

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
    (df["unit_price"] < lower_bound) | (df["unit_price"] > upper_bound), "unit_price"
] = median_price


df["lag_7"] = df["units_sold"].shift(7)
df["lag_14"] = df["units_sold"].shift(14)

df["rolling_mean_7"] = df["units_sold"].rolling(window=7).mean()

df["rolling_std_7"] = df["units_sold"].rolling(window=7).std()

df["rolling_mean_30"] = df["units_sold"].rolling(window=30).mean()

df["rolling_std_30"] = df["units_sold"].rolling(window=30).std()

df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter

df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)

df = df.dropna()

df = pd.get_dummies(df, columns=["product_category", "day_of_week"], drop_first=True)


X = df.drop(
    columns=[
        "units_sold",
        "date",
        "product_id"
    ]
)

scaler = joblib.load("models/scaler.pkl")

X_scaled = scaler.transform(X)


sequence_length = 14

last_sequence = X_scaled[-sequence_length:]

last_sequence = np.expand_dims(last_sequence, axis=0)

input_tensor = torch.tensor(last_sequence, dtype=torch.float32)


input_size = X_scaled.shape[1]

model = LSTMModel(input_size)

model.load_state_dict(torch.load("models/lstm_model.pt"))

model.eval()


with torch.no_grad():

    prediction = model(input_tensor).item()


print("Predicted Demand:", round(prediction, 2))
