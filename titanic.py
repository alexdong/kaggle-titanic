import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

device = torch.device("mps")
train_df = pd.read_csv("datasets/train.csv")
test_df = pd.read_csv("datasets/test.csv")
val_labels = pd.read_csv("datasets/gender_submission.csv")

features = ["Sex", "Pclass", "Age", "Parch", "Fare", "Cabin", "Embarked"]

train_df["Cabin"] = train_df["Cabin"].str[0]
test_df["Cabin"] = test_df["Cabin"].str[0]
label_encoder_cabin = LabelEncoder()
train_df["Cabin"] = label_encoder_cabin.fit_transform(train_df["Cabin"])
test_df["Cabin"] = label_encoder_cabin.fit_transform(test_df["Cabin"])

label_encoder_sex = LabelEncoder()
train_df["Sex"] = label_encoder_sex.fit_transform(train_df["Sex"])
