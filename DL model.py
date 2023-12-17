# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from sklearn.metrics import f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import TimeSeriesSplit




# COMMAND ----------

# Prepare the features as a dense vector column
#assembler = VectorAssembler(inputCols="Features", outputCol="features_dense")
#time_series_df = assembler.transform(time_series_df)

# Extract the necessary columns
#data = time_series_df.select("features_dense", "DEP_DEL15").collect()
#features = [np.array(row.features_dense.toArray()) for row in data]
#labels = [row.DEP_DEL15 for row in data]

# Initialize TimeSeriesSplit for 5 folds
#ts = TimeSeriesSplit(n_splits=5)

# Initialize evaluation lists to store metrics for each fold
f1_scores = []
auc_scores = []

# Iterate through each fold for training and validation
for train, val in final_folds:
    train_features = [np.array(row.Features.toArray()) for row in train]
    train_labels = [row.DEP_DEL15 for row in train]
    val_features = [np.array(row.Features.toArray()) for row in val]
    val_labels = [row.DEP_DEL15 for row in val]

    # Split data into train and validation sets for the current fold
    #train_features, train_labels = np.array(features)[train_index], np.array(features)[val_index]
    #train_labels, val_labels = np.array(labels)[train_index], np.array(labels)[val_index]

    # Convert train and validation data into PyTorch tensors
    train_tensors = torch.tensor(train_features).float()
    train_labels_tensor = torch.tensor(train_labels).float()
    val_tensors = torch.tensor(val_features).float()
    val_labels_tensor = torch.tensor(val_labels).float()

    # Create DataLoader for efficient batching
    train_dataset = TensorDataset(train_tensors, train_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = TensorDataset(val_tensors, val_labels_tensor)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Define LSTM model
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size, hidden_size)
            self.fc = nn.Linear(hidden_size, output_size)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            lstm_out, _ = self.lstm(x.view(len(x), 1, -1))
            output = self.fc(lstm_out.view(len(x), -1))
            return self.sigmoid(output)

    # Instantiate the model, loss function, and optimizer
    model = LSTMModel(input_size=len(train_features[0]), hidden_size=64, output_size=1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    for epoch in range(50):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{50} - Loss: {running_loss / len(train_loader)}")

    # Make predictions on validation data
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in val_loader:
            inputs, _ = data
            outputs = model(inputs)
            predictions.extend(outputs.squeeze().tolist())

    # Calculate AUC and F1 score for the current fold
    auc = roc_auc_score(val_labels, predictions)
    f1 = f1_score(val_labels, np.round(predictions))

    # Append scores to evaluation lists
    auc_scores.append(auc)
    f1_scores.append(f1)

    print(f"AUC: {auc}")
    print(f"F1 Score: {f1}")

    fold_num += 1

# Calculate mean and standard deviation of metrics across folds
mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

print(f"Mean AUC across folds: {mean_auc} ± {std_auc}")
print(f"Mean F1 Score across folds: {mean_f1} ± {std_f1}")

# COMMAND ----------

