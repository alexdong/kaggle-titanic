import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn, optim

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

train_df = pd.read_csv("datasets/train.csv")
test_df = pd.read_csv("datasets/test.csv")

features = ["Sex", "Pclass", "Age", "Parch", "Fare", "Cabin", "Embarked"]
train_df["Cabin"] = train_df["Cabin"].str[0]
test_df["Cabin"] = test_df["Cabin"].str[0]

le_sex = LabelEncoder()
le_cabin = LabelEncoder()
le_embarked = LabelEncoder()

train_df["Sex"] = le_sex.fit_transform(train_df["Sex"])
test_df["Sex"] = le_sex.transform(test_df["Sex"])

all_cabins = pd.concat([train_df["Cabin"], test_df["Cabin"]]).unique()
le_cabin.fit(all_cabins[pd.notna(all_cabins)])
train_df.loc[train_df["Cabin"].notna(), "Cabin"] = le_cabin.transform(
    train_df[train_df["Cabin"].notna()]["Cabin"]
)
test_df.loc[test_df["Cabin"].notna(), "Cabin"] = le_cabin.transform(
    test_df[test_df["Cabin"].notna()]["Cabin"]
)

all_embarked = pd.concat([train_df["Embarked"], test_df["Embarked"]]).unique()
le_embarked.fit(all_embarked[pd.notna(all_embarked)])
train_df.loc[train_df["Embarked"].notna(), "Embarked"] = le_embarked.transform(
    train_df[train_df["Embarked"].notna()]["Embarked"]
)
test_df.loc[test_df["Embarked"].notna(), "Embarked"] = le_embarked.transform(
    test_df[test_df["Embarked"].notna()]["Embarked"]
)

scaler = StandardScaler()
for col in features:
    if col in ["Pclass", "Sex", "Cabin", "Embarked", "Parch"]:
        continue
    train_values = train_df[col][train_df[col].notna()].to_numpy().reshape(-1, 1)
    test_values = test_df[col][test_df[col].notna()].to_numpy().reshape(-1, 1)
    all_values = np.concatenate([train_values, test_values])
    scaler.fit(all_values)
    train_df.loc[train_df[col].notna(), col] = scaler.transform(train_values).flatten()
    if len(test_values) > 0:
        test_df.loc[test_df[col].notna(), col] = scaler.transform(test_values).flatten()


def create_masked_input(
    df: pd.DataFrame, features: list, *, augment: bool = False, mask_prob: float = 0.1
) -> torch.Tensor:
    x = torch.zeros((len(df), len(features) * 2))
    original_mask = torch.zeros((len(df), len(features)))

    for i, col in enumerate(features):
        values = df[col].fillna(0).to_numpy()
        mask = ~df[col].isna().to_numpy()
        x[:, i * 2] = torch.tensor(values, dtype=torch.float32)
        x[:, i * 2 + 1] = torch.tensor(mask, dtype=torch.float32)
        original_mask[:, i] = torch.tensor(mask, dtype=torch.float32)

    if augment:
        aug_mask = torch.rand_like(original_mask) > mask_prob
        aug_mask = aug_mask * original_mask
        for i in range(len(features)):
            x[:, i * 2] = x[:, i * 2] * aug_mask[:, i]
            x[:, i * 2 + 1] = aug_mask[:, i]

    return x


X = create_masked_input(train_df, features)
y = torch.tensor(train_df["Survived"].to_numpy(), dtype=torch.float32).unsqueeze(1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.to(device)
X_val = X_val.to(device)
y_train = y_train.to(device)
y_val = y_val.to(device)

train_indices = torch.arange(len(train_df))


class MLP(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))


model = MLP(len(features) * 2).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = 32
n_epochs = 100

for epoch in range(n_epochs):
    model.train()
    indices = torch.randperm(len(X_train))
    total_loss = 0

    for i in range(0, len(X_train), batch_size):
        batch_indices = indices[i : i + batch_size]
        train_batch_indices = train_indices[batch_indices.cpu()]
        X_batch = create_masked_input(
            train_df.iloc[train_batch_indices], features, augment=True, mask_prob=0.15
        ).to(device)
        y_batch = y_train[batch_indices]

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            threshold = 0.5
            val_acc = ((val_outputs > threshold) == y_val).float().mean()
            print(
                f"Epoch {epoch}, Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

model.eval()
X_test = create_masked_input(test_df, features).to(device)
with torch.no_grad():
    predictions = model(X_test)
    threshold = 0.5
    predictions = (predictions > threshold).int().squeeze().cpu()

submission = pd.DataFrame(
    {"PassengerId": test_df["PassengerId"], "Survived": predictions.numpy()}
)
submission.to_csv("submission.csv", index=False)
print(f"Submission saved. Shape: {submission.shape}")
