# Databricks notebook source
# MAGIC %md
# MAGIC ### Imports

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from collections import namedtuple
from datetime import date, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import re
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, isnan, when, count, split, trim, lit, avg, sum, expr

# COMMAND ----------

# MAGIC %md
# MAGIC ### Connect to Storage

# COMMAND ----------

blob_container  = "261container"        # The name of your container created in https://portal.azure.com
storage_account = "261bucket"  # The name of your Storage account created in https://portal.azure.com
secret_scope    = "261scope"           # The name of the scope created in your local computer using the Databricks CLI
secret_key      = "scopekey"             # The name of the secret key created in your local computer using the Databricks CLI
team_blob_url   = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"  #points to the root of your team storage bucket
mids261_mount_path      = "/mnt/mids-w261"

# SAS Token: Grant the team limited access to Azure Storage resources
spark.conf.set(
    f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
    dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# Print what's saved in blog storage
display(dbutils.fs.ls(f"{team_blob_url}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Helper Functions

# COMMAND ----------

# Function that modifies folds to only include features of interest

from pyspark.ml.feature import VectorSlicer

def final_feature_finder(folds, indices):
    """Parameters:
    - folds (list): A list of tuples, where each tuple represents a fold containing
                   a training set and a validation set.
    - indices (list): A list of features where each element is the index for the feature you want to include

    Returns:
    list: A new list of folds, where each fold includes the original training and
          validation sets augmented with additional features based on hard coded LASSO coefficients.
    """

    slicer = VectorSlicer(inputCol="scaledFeatures", outputCol="Features", indices=indices)

    def feature_create(df):
    
        output = slicer.transform(df)
        return output
    
    new_folds = []
    for train,val in folds:
        new_train = feature_create(train)
        new_val = feature_create(val)
        new_folds.append((new_train,new_val))
    return new_folds


# COMMAND ----------

# Function that takes a single model's output takes the average of F1 and AUC scores across folds and takes the average of a model's F1 and AUC scores across seeds

def average_scores(metrics):
    """
    Takes the average F1 and AUC scores witin each seed (i.e. across folds) and then takes the average across seeds
    
    Input E.g.:
    {'Seed #5': [[0.8307, 0.1562], [0.4738, 0.0998]], 
    'Seed #33': [[0.4387, 0.7443], [0.1616, 0.2837]]}
    
    Output: Average F1 score across seeds, Average AUC score across seeds"""

    # Take average within each seed
    # Initialize scores dictionary
    seed_scores = {}

    # Iterate through each seed
    for key, value in metrics.items():
        f1_scores = 0
        auc_scores = 0
        
        # Iterate through the folds within that seed
        for scores in value:
            f1_scores += scores[0]
            auc_scores += scores[1]
        
        # Take the average scores per seed
        # DELETE updated this to use weighted averages
        f1_avg = f1_scores/len(value)
        auc_avg = auc_scores/len(value)
            
        seed_scores[key] = [f1_avg, auc_avg]

        print(f"Average F1 Score within {key}: {f1_avg}")
        print(f"Average AUC Score within {key}: {auc_avg}")


    # Take average across seeds
    # Initialize final values
    final_f1_scores = 0
    final_auc_scores = 0

    # Iterate through each seed
    for key, value in seed_scores.items():
        final_f1_scores += value[0]
        final_auc_scores += value[1]

    f1_avg_final = final_f1_scores/len(seed_scores)
    auc_avg_final = final_auc_scores/len(seed_scores)

    print(f"Average F1 Score Across Seeds: {f1_avg_final}")
    print(f"Average AUC Score Across Seeds: {auc_avg_final}")

    return (f1_avg_final, auc_avg_final)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Data

# COMMAND ----------

# Load full data

test_df = spark.read.parquet(f"{team_blob_url}/test_transformed_2023_12_10").cache()

val_df = spark.read.parquet(f"{team_blob_url}/val_transformed_2023_12_10").cache()

def load_folds_from_blob_and_cache(blob_url, fold_name):
    folds = list()

    # Compute the fold count
    files = dbutils.fs.ls(f"{blob_url}/{fold_name}")
    fold_names = sorted([f.name for f in files if f.name.startswith("train")])
    match = re.match(r"train_(\d+)_df", fold_names[-1])
    fold_count = int(match.group(1)) + 1
    print(f"Loading {fold_count} folds...")

    # Load folds
    for i in range(fold_count):
        train_df = (
            spark.read.parquet(f"{blob_url}/{fold_name}/train_{i}_df")
            .cache()
        )
        val_df = (
            spark.read.parquet(f"{blob_url}/{fold_name}/val_{i}_df")
            .cache()
        )
        folds.append((train_df, val_df))
    return folds

folds = load_folds_from_blob_and_cache(team_blob_url, 'k_folds_cross_val_2023_12_10')

# COMMAND ----------

new_folds = []
for fold in folds:
    df1_modified = fold[0].drop('date_index', 'log_date_index', 'normalized_log_weight')
    df2_modified = fold[1].drop('date_index', 'log_date_index', 'normalized_log_weight')

    new_folds.append((df1_modified, df2_modified))

folds = new_folds

# COMMAND ----------

# Check output
print((folds[0][1].display(1)))

# COMMAND ----------

# Create subset of data

train_fold0, val_fold0 = folds[0]
fold0 = [folds[0]]
fold0

# COMMAND ----------

# See feature names and indices

# I think metadata we need gets dropped when we do StandardScaler in the original pipeline. StandardScaler produces scaledFeatures which doesn't contain metadata. 
# I don't think we need to map scaledFeatures back to original features. They look the same according to block 50 in Phase II EDA & Model Prep
# Easiest thing might just be a manual mapping once we have our features figured out

# train_fold0.schema['Features'].metadata
# train_fold0.schema['scaledFeatures'].metadata

pd.set_option('display.max_rows', None)

pd.DataFrame(train_fold0.schema["features"].metadata["ml_attr"]["attrs"]["binary"]+train_fold0.schema["features"].metadata["ml_attr"]["attrs"]["numeric"]).sort_values("idx")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Logistic Regression & LASSO Regularization

# COMMAND ----------

# MAGIC %md Define Functions

# COMMAND ----------

# Define Logistic Regression Function

def trainLogRegModel(folds, label, feature, final_val, seeds):
    trainingMetrics = {}
    validationMetrics = {}
    finalValidationMetrics = {}

    # Run for multiple seeds    
    for seed in seeds:
        seed_name = f"Seed #{seed}" 
        trainingMetrics[seed_name] = []
        validationMetrics[seed_name] = []
        finalValidationMetrics[seed_name] = []
        fold_num = 0

        # Create a Logistic Regression model instance
        lr = LogisticRegression(labelCol=label, featuresCol=feature, weightCol = 'combined_weight')

        # Evaluation metrics
        # DELETE see if can move these outside of the loop
        evaluator_f1 = MulticlassClassificationEvaluator(labelCol=label, metricName="f1")
        evaluator_auc = BinaryClassificationEvaluator(labelCol=label, metricName="areaUnderROC")

        for fold in folds:
            train_df, val_df = fold
            
            # Fit the model on the training data
            model = lr.fit(train_df)

            # Print model coeficients
            print(f"Seed #{seed} Fold #{fold_num} Coefficients: {model.coefficients}")
            fold_num += 1

            # Make predictions on the training data and evaluate
            train_predictions = model.transform(train_df)
            trainingMetrics[seed_name].append([evaluator_f1.evaluate(train_predictions), evaluator_auc.evaluate(train_predictions)])

            # Make predictions on the validation data and evaluate
            val_predictions = model.transform(val_df)
            validationMetrics[seed_name].append([evaluator_f1.evaluate(val_predictions), evaluator_auc.evaluate(val_predictions)])

            if (final_val != None):
                final_val_predictions = model.transform(final_val)
                finalValidationMetrics[seed_name].append([evaluator_f1.evaluate(final_val_predictions), evaluator_auc.evaluate(final_val_predictions)])

    return (trainingMetrics, validationMetrics, finalValidationMetrics)

# COMMAND ----------

# Define LASSO Function

def trainLassoRegModel(folds, label, feature, final_val, seeds):
    trainingMetrics = {}
    validationMetrics = {}
    finalValidationMetrics = {}

    # Run for multiple seeds    
    for seed in seeds:
        seed_name = f"Seed #{seed}" 
        trainingMetrics[seed_name] = []
        validationMetrics[seed_name] = []
        finalValidationMetrics[seed_name] = []
        fold_num = 0

        # Create a Logistic Regression model instance
        lr = LogisticRegression(labelCol=label, featuresCol=feature, weightCol = 'combined_weight', regParam =0.01, elasticNetParam=1.0)

        # Evaluation metrics
        # DELETE see if can move these outside of the loop
        evaluator_f1 = MulticlassClassificationEvaluator(labelCol=label, metricName="f1")
        evaluator_auc = BinaryClassificationEvaluator(labelCol=label, metricName="areaUnderROC")

        for fold in folds:
            train_df, val_df = fold
            
            # Fit the model on the training data
            model = lr.fit(train_df)

            # Print model coeficients
            print(f"Seed #{seed} Fold #{fold_num} Coefficients: {model.coefficients}")
            fold_num += 1

            # Make predictions on the training data and evaluate
            train_predictions = model.transform(train_df)
            trainingMetrics[seed_name].append([evaluator_f1.evaluate(train_predictions), evaluator_auc.evaluate(train_predictions)])

            # Make predictions on the validation data and evaluate
            val_predictions = model.transform(val_df)
            validationMetrics[seed_name].append([evaluator_f1.evaluate(val_predictions), evaluator_auc.evaluate(val_predictions)])

            if (final_val != None):
                final_val_predictions = model.transform(final_val)
                finalValidationMetrics[seed_name].append([evaluator_f1.evaluate(final_val_predictions), evaluator_auc.evaluate(final_val_predictions)])

    return (trainingMetrics, validationMetrics, finalValidationMetrics)

# COMMAND ----------

# MAGIC %md Modeling

# COMMAND ----------

# Run logistic regression on all 70 features on subset of data
# With 1 seed and val cut took 12 minutes
seeds = [5]

# DELETE run this with scaledFeatures instead of features if redo
initlog_trainingMetrics0, initlog_validationMetrics0, initlog_ValMetrics0 = trainLogRegModel(fold0, "DEP_DEL15", 'features', val_df, seeds)

# COMMAND ----------

# Run logistic regression on all 70 features on all folds
# With 1 seed and val cut took 24-32 minutes
seeds = [5]

initlog_trainingMetrics, initlog_validationMetrics, initlog_ValMetrics = trainLogRegModel(folds, "DEP_DEL15", 'scaledFeatures', val_df, seeds)

# COMMAND ----------

# Run LASSO on all 70 features on subset of data
# With 1 seed and val cut took 11 minutes
seeds = [5]

# DELETE run this with scaledFeatures instead of features if redo
# DELETE ignore these results
lasso_trainingMetrics0, lasso_validationMetrics0, lasso_ValMetrics0 = trainLassoRegModel(fold0, "DEP_DEL15", 'features', val_df, seeds)

# COMMAND ----------

# Take out derived features for LASSO. Will add them back in for later models
# Dropping YEAR also because won't hold valuable info for future data

# 35 - pagerank
# 63 - previous_flight_delay
# 64 - days_to_nearest_holiday
# 67 - Plane_Delays_last_24H
# 68 - airport_delays_in_previous_24_hours
# 69 - YEAR

folds_noderived = final_feature_finder(folds, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,65,66])


# 36 becomes 35
# 65 becomes 62 (because 3 have been removed by that point)
print((folds_noderived[0][0].display(1)))

# COMMAND ----------

# Create validation set without derived features

noderived_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,65,66]

slicer = VectorSlicer(inputCol="scaledFeatures", outputCol="Features", indices=noderived_indices)

val_noderived = slicer.transform(val_df)


# print(val_noderived.display(1))

# COMMAND ----------

# Run LASSO on 64 features (excluding derived) on all folds
# With 1 seed and val cut took 28 minutes
seeds = [77]

lasso_trainingMetrics, lasso_validationMetrics, lasso_ValMetrics = trainLassoRegModel(folds_noderived, "DEP_DEL15", 'Features', val_noderived, seeds)

# COMMAND ----------

# Identify features that were not dropped to 0 in at least 1 fold

lasso0 = [0,1,4,7,10,15,18,21,30,34,38,44,57,61,62,63]
lasso1 = [0,1,6,7,10,15,19,34,37,39,41,44,51,54,57,61,62,63]
lasso2 = [0,1,3,6,7,13,15,20,22,24,25,32,34,39,41,44,51,57,61,62,63]
lasso3 = [0,1,6,7,15,17,19,20,21,27,30,33,34,39,40,44,48,51,54,57,61,62,63]
lasso4 = [0,1,6,7,12,13,15,32,33,36,41,44,51,57,61,62,63]

lasso_full = []

for i in range(64):
    if (i in lasso0) or (i in lasso1) or (i in lasso2) or (i in lasso3) or (i in lasso4):
        lasso_full.append(i)

print(lasso_full)
print(f"{len(lasso_full)} features were retained by at least 1 fold")

# COMMAND ----------

# Create folds and validation set with 37 features that were retained by LASSO + 5 derived

# Have to map some features back to original scaledFeatures indices
# Do not add 69 = YEAR back in
# Adding 36 = dest_airport_lat back in even tho dropped by LASSO
# +1 to features between 35-61 inclusive
# +3 to features between 62-63 inclusive
lasso_indices = [0, 1, 3, 4, 6, 7, 10, 12, 13, 15, 17, 18, 19, 20, 21, 22, 24, 25, 27, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 45, 49, 52, 55, 58, 62, 63, 64, 65, 66, 67, 68]

folds_lasso = final_feature_finder(folds, lasso_indices)

slicer_lasso = VectorSlicer(inputCol="scaledFeatures", outputCol="Features", indices=lasso_indices)

val_lasso = slicer_lasso.transform(val_df)


# COMMAND ----------

# Run logistic regression on 38 lasso + 5 derived features on all folds
# With 1 seed and val cut took 30 minutes
seeds = [5]

lassolog_trainingMetrics, lassolog_validationMetrics, lassolog_ValMetrics = trainLogRegModel(folds_lasso, "DEP_DEL15", 'Features', val_lasso, seeds)

# COMMAND ----------

# MAGIC %md Evaluation

# COMMAND ----------

average_scores(initlog_trainingMetrics)

# COMMAND ----------

average_scores(lassolog_trainingMetrics)

# COMMAND ----------

average_scores(initlog_validationMetrics)

# COMMAND ----------

average_scores(lassolog_validationMetrics)

# COMMAND ----------

average_scores(initlog_ValMetrics)

# COMMAND ----------

average_scores(lassolog_ValMetrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLP

# COMMAND ----------

print(len(final_folds[0][0].select('Features').first()[0]))

# COMMAND ----------

# Train a MLP Neural Network - Extra Credit (Up to 5 points) if you are able to run other NN or Deep Learning technique
# -- Implement a multilayer perceptron (MLP)  Neural Network (NN) model
# -- Experiment with at least 2 different MLP Network architectures (e.g., say, one with one hidden layer and a second with two hidden layers).
# -- Report neural network architecture in string form (e.g., MLP-10 - Relu - 2 Softmax )

# For more background and a template pipeline, please see here:
# https://spark.apache.org/docs/latest/ml-classification-regression.html#multilayer-perceptron-classifierLinks to an external site.


def MLP(label, layers, folds, final_val):
    trainingMetrics = []
    validationMetrics = []
    finalValidationMetrics = []

    # create the trainer and set its parameters
    trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234, labelCol=label, featuresCol = 'Features')

   

    evaluator_f1 = MulticlassClassificationEvaluator(labelCol=label, metricName="f1")
    evaluator_auc = BinaryClassificationEvaluator(labelCol=label, metricName="areaUnderROC")

    for train, val in folds:
         # train the model
        model = trainer.fit(train)

        # Make predictions on the training data and evaluate
        train_predictions = model.transform(train)
        trainingMetrics.append([evaluator_f1.evaluate(train_predictions), evaluator_auc.evaluate(train_predictions)])

        val_predictions = model.transform(val)
        validationMetrics.append([evaluator_f1.evaluate(val_predictions), evaluator_auc.evaluate(val_predictions)])

        if (final_val != None):
            val_predictions = model.transform(final_val)
            finalValidationMetrics.append([evaluator_f1.evaluate(val_predictions), evaluator_auc.evaluate(val_predictions)])
            
    return (trainingMetrics, validationMetrics, finalValidationMetrics)


# COMMAND ----------

#2 Hidden layers MLP

layers = [len(final_folds[0][0].select('Features').first()[0]), 17, 11, 2]
trainingMetrics, validationMetrics, finalValidationMetrics = MLP('DEP_DEL15', layers, final_folds, val_df)

print("MLP 2 hidden layers Training Metrics:", trainingMetrics)
print("MLP 2 hidden layers Validation Metrics:", validationMetrics)
print("MLP 2 hidden layers final Validation Metrics:", finalValidationMetrics)

# COMMAND ----------

#2 Hidden layers MLP

layers = [len(folds_lasso[0][0].select('Features').first()[0]), 17, 11, 2]
trainingMetrics, validationMetrics, finalValidationMetrics = MLP('DEP_DEL15', layers, folds_lasso, val_lasso)

print("MLP 2 hidden layers Training Metrics:", trainingMetrics)
print("MLP 2 hidden layers Validation Metrics:", validationMetrics)
print("MLP 2 hidden layers final Validation Metrics:", finalValidationMetrics)

# COMMAND ----------

# 1 Hidden layer MLP 
layers = [len(folds_lasso[0][0].select('Features').first()[0]), 17, 2]
trainingMetrics, validationMetrics, finalValidationMetrics = MLP('DEP_DEL15', layers, folds_lasso, val_lasso)

print("MLP 1 hidden layer Training Metrics:", trainingMetrics)
print("MLP 1 hidden layer Validation Metrics:", validationMetrics)
print("MLP 1 hidden layer Validation Metrics:", finalValidationMetrics)

# COMMAND ----------

# MAGIC %md
# MAGIC LSTM modeling (DL)

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors
from sklearn.metrics import f1_score, roc_auc_score
from pyspark.ml.feature import VectorAssembler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors, DenseVector
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType

# Initialize evaluation lists to store metrics for each fold
f1_scores= []
auc_scores= []
training_fold_metrics = []
val_fold_metrics=[]
final_val_metrics=[]

# Define a UDF to convert SparseVector to DenseVector
def sparse_to_list(sv):
    return [float(x) for x in sv.toArray()]

# Convert the SparseVector column to a DenseVector column using the UDF
sparse_to_list_udf = udf(sparse_to_list, returnType=ArrayType(DoubleType()))

# Pick 10% rows of data
final_val = val_lasso.sample(False, 0.1, seed=123)

model, criterion, optimizer = None, None, None

# Convert the SparseVector column to a DenseVector column using the UDF
final_val = final_val.withColumn("dense_features", sparse_to_list_udf("Features"))
# Collect the values of the `dense_features` column into a list
dense_features_list = final_val.select("dense_features").collect()
# Convert the collected values to NumPy arrays and store them in a list
final_val_features = np.array([np.array(row["dense_features"]) for row in dense_features_list])
final_val_labels = np.array([row["DEP_DEL15"] for row in final_val.select("DEP_DEL15").collect()])

# Convert final validation data into PyTorch tensors
final_val_tensors = torch.tensor(final_val_features).float()
final_val_labels_tensor = torch.tensor(final_val_labels).float()

final_val_dataset = TensorDataset(final_val_tensors, final_val_labels_tensor)
final_val_loader = DataLoader(final_val_dataset, batch_size=64, shuffle=False)

fold_num=0
# Iterate through each fold for training and validation
for train, val in folds_lasso:
    fold_num += 1
    print(f'Fold num: {fold_num}')
    # Pick 10% rows of data
    train = train.sample(False, 0.1, seed=123)
    val = val.sample(False, 0.1, seed=123)

    # Convert the SparseVector column to a DenseVector column using the UDF
    train = train.withColumn("dense_features", sparse_to_list_udf("Features"))
    # Collect the values of the `dense_features` column into a list
    dense_features_list = train.select("dense_features").collect()
    # Convert the collected values to NumPy arrays and store them in a list
    train_features = np.array([np.array(row["dense_features"]) for row in dense_features_list])
    train_labels = [row["DEP_DEL15"] for row in train.select("DEP_DEL15").collect()]
    
    # Convert the SparseVector column to a DenseVector column using the UDF
    val = val.withColumn("dense_features", sparse_to_list_udf("Features"))
    # Collect the values of the `dense_features` column into a list
    dense_features_list = val.select("dense_features").collect()
    # Convert the collected values to NumPy arrays and store them in a list
    val_features = np.array([np.array(row["dense_features"]) for row in dense_features_list])
    val_labels = [row["DEP_DEL15"] for row in val.select("DEP_DEL15").collect()]

    # Convert train and validation data into PyTorch tensors
    train_tensors = torch.tensor(train_features).float()
    train_labels_tensor = torch.tensor(train_labels).float()
    
    val_tensors = torch.tensor(val_features).float()
    val_labels_tensor = torch.tensor(val_labels).float()

    # Create DataLoaders for efficient batching
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
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = data
            labels = labels.view(-1, 1) if labels.dim() == 1 else labels  # Reshape labels to match model output size
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{10} - Loss: {running_loss / len(train_loader)}")

    # Make predictions on validation data of each fold
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
    val_fold_metrics.append([f1,auc])

    # Append scores to evaluation lists
    auc_scores.append(auc)
    f1_scores.append(f1)
    
    print(f"Fold{fold_num} F1 Score: {f1}")
    print(f"Fold{fold_num} AUC: {auc}")


# Make predictions on final validation data 
model.eval()
predictions = []
with torch.no_grad():
    for data in final_val_loader:
        inputs, _ = data
        outputs = model(inputs)
        predictions.extend(outputs.squeeze().tolist())

# Calculate AUC and F1 score for the current fold
final_val_auc = roc_auc_score(final_val_labels, predictions)
final_val_f1 = f1_score(final_val_labels, np.round(predictions))

final_val_metrics.append([final_val_f1,final_val_auc])

print(f"Final Val F1 Score: {final_val_f1}")
print(f" Final Val AUC: {final_val_auc}")

# Calculate mean and standard deviation of metrics across folds
mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

print(f"Mean AUC across folds: {mean_auc} ± {std_auc}")
print(f"Mean F1 Score across folds: {mean_f1} ± {std_f1}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random Forest

# COMMAND ----------

# MAGIC %md Notes
# MAGIC
# MAGIC - Documentation: https://spark.apache.org/docs/3.1.3/api/python/reference/api/pyspark.ml.classification.RandomForestClassifier.html
# MAGIC - While we should pick a numTree value based on compute/runtime, I don't think we need to iterate through different numTree arguments for performance since more will hypothetically always be better
# MAGIC - Automatically uses bootstrap samples
# MAGIC - featureSubsetStrategy
# MAGIC   - Chooses "auto" by default
# MAGIC   - 'all' (use all features), 'onethird' (use 1/3 of the features), 'sqrt' (use sqrt(number of features)), 'log2' (use log2(number of features)), 'n' (when n is in the range (0, 1.0], use n * number of features.
# MAGIC   - Val F1 is slightly higher with auto select than using all
# MAGIC - impurity 
# MAGIC   - uses gini by default, could try entropy
# MAGIC - maxDepth 
# MAGIC   - auto set to 5
# MAGIC   - maximum depth of tree
# MAGIC   - 10 depth did about the same as 5 depth with same amount of trees
# MAGIC - numTrees
# MAGIC   - 10 trees did better than 5 by a little bit
# MAGIC   - 15 trees took only a few more minutes than 10 trees and about the same as 5
# MAGIC   - 15 trees had slightly lower F1 than 10 trees
# MAGIC - Runtimes are pretty consistently 15-20 minutes with 3 seeds while changing hyperparameters
# MAGIC
# MAGIC - Best hyperparameters on subset
# MAGIC   - featureSubsetStrategy = auto
# MAGIC   - maxDepth = 5
# MAGIC   - numTrees = 10

# COMMAND ----------

# MAGIC %md Define Random Forest Function

# COMMAND ----------

def trainRandomForestModel(folds, label, feature, final_val, numTrees, seeds, featureSubsetStrategy, maxDepth):
    trainingMetrics = {}
    validationMetrics = {}
    finalValidationMetrics = {}

    # Run for multiple seeds    
    for seed in seeds:
        seed_name = f"Seed #{seed}" 
        trainingMetrics[seed_name] = []
        validationMetrics[seed_name] = []
        finalValidationMetrics[seed_name] = []
        fold_num = 0

        # Create a Random Forest model instance
        rf = RandomForestClassifier(labelCol=label, featuresCol=feature, numTrees=numTrees, weightCol='combined_weight', seed=seed, featureSubsetStrategy=featureSubsetStrategy, maxDepth=maxDepth)

        # Evaluation metrics
        # DELETE see if still works if move this outside the loop
        evaluator_f1 = MulticlassClassificationEvaluator(labelCol=label, metricName="f1")
        evaluator_auc = BinaryClassificationEvaluator(labelCol=label, metricName="areaUnderROC")

        # Fit model to each fold
        for fold in folds:
            train_df, val_df = fold

            model = rf.fit(train_df)

            # Access feature importances
            # DELETE confirm this is sufficient for what we need before running full model
            feature_importances = model.featureImportances
            print(f"Seed #{seed} Fold #{fold_num} Feature Importances: {feature_importances}")
            fold_num += 1

            # Make predictions on the training data and evaluate
            train_predictions = model.transform(train_df)
            trainingMetrics[seed_name].append([evaluator_f1.evaluate(train_predictions), evaluator_auc.evaluate(train_predictions)])

            # Make predictions on the validation data and evaluate
            val_predictions = model.transform(val_df)
            validationMetrics[seed_name].append([evaluator_f1.evaluate(val_predictions), evaluator_auc.evaluate(val_predictions)])

            if (final_val != None):
                final_val_predictions = model.transform(final_val)
                finalValidationMetrics[seed_name].append([evaluator_f1.evaluate(final_val_predictions), evaluator_auc.evaluate(final_val_predictions)])
    
    return (trainingMetrics, validationMetrics, finalValidationMetrics)

# COMMAND ----------

# MAGIC %md Subset Modeling and Hyperparameter Testing

# COMMAND ----------

# Times with 1 seed - 8 minutes 3 trees, 9 minutes 1 tree, 9 minutes 2 trees
# featureSubsetStrategy = auto
seeds = [5]

trainMetrics_fold0, valMetrics_fold0 = trainRandomForestModel(fold0, 'DEP_DEL15', 'Features', 2, seeds)

# COMMAND ----------

trainMetrics_fold0

# COMMAND ----------

# Time with 2 seeds, 1 tree - 10 minutes 
# Time with 3 seeds, 1 tree - 19 minutes
# Time with 3 seeds, 5 trees - 18 minutes
# featureSubsetStrategy = auto
seeds = [5, 33, 12]

trainMetrics_fold0_auto, valMetrics_fold0_auto = trainRandomForestModel(fold0, 'DEP_DEL15', 'Features', 5, seeds, "auto")

# COMMAND ----------

average_scores(trainMetrics_fold0_auto)

# COMMAND ----------

average_scores(valMetrics_fold0_auto)

# COMMAND ----------

# Time with 3 seeds, 5 trees - 18 minutes
# featureSubsetStrategy = all

# Runtime with auto and all is about the same
# Val F1 is slightly higher with auto select than using all

seeds = [5, 33, 12]

trainMetrics_fold0_all, valMetrics_fold0_all = trainRandomForestModel(fold0, 'DEP_DEL15', 'Features', 5, seeds, "all")

# COMMAND ----------

average_scores(trainMetrics_fold0_all)

# COMMAND ----------

average_scores(valMetrics_fold0_all)

# COMMAND ----------

# Time with 3 seeds, 10 trees - 14 minutes

# 10 trees did better than 5 by a little bit

seeds = [5, 33, 12]

trainMetrics_fold0_10, valMetrics_fold0_10 = trainRandomForestModel(fold0, 'DEP_DEL15', 'Features', 10, seeds, "auto")

# COMMAND ----------

average_scores(valMetrics_fold0_10)

# COMMAND ----------

# Time with 3 seeds, 15 trees - 17 minutes

# 15 trees took only a few more minutes than 10 trees and about the same as 5
# 15 trees had slightly lower F1 than 10 trees

seeds = [5, 33, 12]

trainMetrics_fold0_15, valMetrics_fold0_15 = trainRandomForestModel(fold0, 'DEP_DEL15', 'Features', 15, seeds, "auto")

# COMMAND ----------

average_scores(valMetrics_fold0_15)

# COMMAND ----------

# Time with 3 seeds, 10 trees, 10 depth - 20 minutes

# 10 depth did about the same as 5 depth with same amount of trees

seeds = [5, 33, 12]

trainMetrics_fold0_deep, valMetrics_fold0_deep = trainRandomForestModel(fold0, 'DEP_DEL15', 'Features', 10, seeds, "auto", 10) 

# COMMAND ----------

average_scores(valMetrics_fold0_deep)

# COMMAND ----------

# MAGIC %md Full Modeling

# COMMAND ----------

# Run Random Forest on 38 lasso + 5 derived features on all folds
# With 1 seed and val cut took X minutes
seeds = [28]

rf_trainingMetrics, rf_validationMetrics, rf_ValMetrics = trainRandomForestModel(folds_lasso, "DEP_DEL15", 'Features', val_lasso, 10, seeds, "auto", 5)

# COMMAND ----------

# MAGIC %md Evaluation

# COMMAND ----------

average_scores(rf_trainingMetrics)

# COMMAND ----------

average_scores(rf_validationMetrics)

# COMMAND ----------

average_scores(rf_ValMetrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gradient Boosted Decision Trees

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Notes
# MAGIC - Documentation: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.GBTClassifier.html
# MAGIC - Determine maxIter value
# MAGIC - maxDepth
# MAGIC   - Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. Must be in range [0, 30].'
# MAGIC - featureSubsetStrategy
# MAGIC   - set to 'sqrt' for classification and to 'onethird' for regression), 'all' (use all features)...
# MAGIC   - "all" by default
# MAGIC   - All got the same scores as when auto was specified. Auto must be using All
# MAGIC - stepSize 
# MAGIC   - 'Step size (a.k.a. learning rate) in interval (0, 1] for shrinking the contribution of each estimator.'
# MAGIC   - default = 0.1
# MAGIC   - stepSize 0.05 did about the same as 0.1
# MAGIC - maxIter
# MAGIC   - default = 20
# MAGIC   - 20 score was about the same as 10
# MAGIC - Uses logistic loss

# COMMAND ----------

# MAGIC %md Define GBDT Function

# COMMAND ----------

def trainGBDTModel(folds, label, feature, final_val, seeds, maxIter=10, stepSize=0.1, featureSubsetStrategy='all'):
    trainingMetrics = {}
    validationMetrics = {}
    finalValidationMetrics = {}

    # Run for multiple seeds    
    for seed in seeds:
        seed_name = f"Seed #{seed}" 
        trainingMetrics[seed_name] = []
        validationMetrics[seed_name] = []
        finalValidationMetrics[seed_name] = []
        fold_num = 0

        # Create GBDT model instance
        gbt = GBTClassifier(labelCol=label, featuresCol=feature, weightCol='combined_weight', seed=seed, maxIter=maxIter, stepSize=stepSize, featureSubsetStrategy=featureSubsetStrategy)

        # Create a Pipeline
        pipeline = Pipeline(stages=[gbt])

        # Evaluation metrics
        # DELETE see if still works if move this outside the loop
        evaluator_f1 = MulticlassClassificationEvaluator(labelCol='DEP_DEL15', metricName="f1")
        evaluator_auc = BinaryClassificationEvaluator(labelCol='DEP_DEL15', metricName="areaUnderROC")

        # Fit model to each fold
        for fold in folds:
            train_df, val_df = fold

            # Train the model
            model = pipeline.fit(train_df)

            # Access feature importances
            # DELETE confirm this is sufficient for what we need before running full model
            # DELETE confirm pulling correct thing
            feature_importances = model.stages[-1].featureImportances
            print(f"Seed #{seed} Fold #{fold_num} Feature Importances: {feature_importances}")
            fold_num += 1

            # Make predictions on the training data and evaluate
            train_predictions = model.transform(train_df)
            trainingMetrics[seed_name].append([evaluator_f1.evaluate(train_predictions), evaluator_auc.evaluate(train_predictions)])

            # Make predictions on the validation data and evaluate
            val_predictions = model.transform(val_df)
            validationMetrics[seed_name].append([evaluator_f1.evaluate(val_predictions), evaluator_auc.evaluate(val_predictions)])

            if (final_val != None):
                final_val_predictions = model.transform(final_val)
                finalValidationMetrics[seed_name].append([evaluator_f1.evaluate(final_val_predictions), evaluator_auc.evaluate(final_val_predictions)])

    return (trainingMetrics, validationMetrics, finalValidationMetrics)



# COMMAND ----------

# MAGIC %md Subset Modeling and Hyperparameter Testing

# COMMAND ----------

# maxIter=10 took 9 minutes, 19 minutes

seeds = [5, 33, 12]

gbdt_trainMetrics_fold0_10, gbdt_valMetrics_fold0_10 = trainGBDTModel(fold0, 'DEP_DEL15', 'Features', seeds, maxIter=10)

# COMMAND ----------

average_scores(gbdt_valMetrics_fold0_10)

# COMMAND ----------

# maxIter=20 took 22 minutes

# 20 performed about the same as 10 iterations

seeds = [5, 33, 12]

gbdt_trainMetrics_fold0_20, gbdt_valMetrics_fold0_20 = trainGBDTModel(fold0, 'DEP_DEL15', 'Features', seeds, maxIter=20)

# COMMAND ----------

average_scores(gbdt_valMetrics_fold0_20)

# COMMAND ----------

# stepSize=0.05 took 17 minutes

# stepSize 0.05 did about the same as 0.1

seeds = [5, 33, 12]

gbdt_trainMetrics_fold0_smstep, gbdt_valMetrics_fold0_smstep = trainGBDTModel(fold0, 'DEP_DEL15', 'Features', seeds, maxIter=10, stepSize=0.05)

# COMMAND ----------

average_scores(gbdt_valMetrics_fold0_smstep)

# COMMAND ----------

# auto took 19 minutes

# Auto did the exact same as default. See test below

seeds = [5, 33, 12]

gbdt_trainMetrics_fold0_auto, gbdt_valMetrics_fold0_auto = trainGBDTModel(fold0, 'DEP_DEL15', 'Features', seeds, maxIter=10, stepSize=0.1, featureSubsetStrategy='auto')

# COMMAND ----------

average_scores(gbdt_valMetrics_fold0_auto)

# COMMAND ----------

# all took 18 minutes

# All got the same scores as when auto was specified. Auto must be using All

seeds = [5, 33, 12]

# NOTE - Forgot to change this name. Update if run again
gbdt_trainMetrics_fold0_auto, gbdt_valMetrics_fold0_auto = trainGBDTModel(fold0, 'DEP_DEL15', 'Features', seeds, maxIter=10, stepSize=0.1, featureSubsetStrategy='all')

# COMMAND ----------

average_scores(gbdt_valMetrics_fold0_auto)

# COMMAND ----------

# MAGIC %md Full Modeling

# COMMAND ----------

# Run GBDT on 38 lasso + 5 derived features on all folds
seeds = [66]

gbdt_trainingMetrics, gbdt_validationMetrics, gbdt_ValMetrics = trainGBDTModel(folds_lasso, "DEP_DEL15", 'Features', val_lasso, seeds, 10, 0.1, "auto")

# COMMAND ----------

# MAGIC %md Evaluation

# COMMAND ----------

average_scores(gbdt_trainingMetrics)

# COMMAND ----------

average_scores(gbdt_validationMetrics)

# COMMAND ----------

average_scores(gbdt_ValMetrics)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Final Modeling

# COMMAND ----------

# MAGIC %md *Create Data*

# COMMAND ----------

# Stitch together data for full training set

# 3968611 rows in the first train fold
# 2012417 rows in the last val fold
# 1931041 rows in the big val cut

# Join all the training folds - 19911704 rows
train_df = folds[0][0].union(folds[1][0]).union(folds[2][0]).union(folds[3][0]).union(folds[4][0])

# Join the last val fold - 21924121 rows
train_df = train_df.union(folds[4][1])

# Filter to only columns of interest (otherwise merging issue with val_df)
# Comment these lines out if don't want to include val_df in training data
train_df = train_df.select(['DEP_DEL15', 'scaledFeatures', 'FL_DATE'])
val_df_modified = val_df.select(['DEP_DEL15', 'scaledFeatures', 'FL_DATE'])

# Join the big validation cut - 23855162 rows
# Comment this line out if don't want to include val_df in training data
train_df = train_df.union(val_df_modified)

# Drop duplicates - 23853421 rows
train_df = train_df.dropDuplicates()

# COMMAND ----------

# Recompute the combined_weight column on the new training data
# NOTE Don't run this unless including val_df in training data

# Compute class weights
num_negatives = train_df.filter(train_df['DEP_DEL15'] == 0).count()
total = train_df.count()
weight_ratio = num_negatives / total

# Add weights
train_df = train_df.withColumn('weight', when(train_df['DEP_DEL15'] == 1, weight_ratio).otherwise(1 - weight_ratio))

# Find the minimum date to establish a reference point
min_date = train_df.agg({"FL_DATE": "min"}).collect()[0][0]
train_df = train_df.withColumn("date_index", F.datediff(F.col("FL_DATE"), F.lit(min_date)))

# Apply logarithmic transformation
train_df = train_df.withColumn("log_date_index", F.log(F.col("date_index") + 1))  # Adding 1 to avoid log(0)

# Compute the total sum of the log_date_index column
total_log_date_index = train_df.agg(F.sum("log_date_index")).collect()[0][0]

max_log_date_index = train_df.agg(F.max("log_date_index")).collect()[0][0]
min_log_date_index = train_df.agg(F.min("log_date_index")).collect()[0][0]

# Add normalized weights
train_df = train_df.withColumn('normalized_log_weight', 
                               (F.col('log_date_index') - min_log_date_index) /(max_log_date_index - min_log_date_index) * 0.7 + 0.3
                               )

train_df = train_df.withColumn('combined_weight', F.col('weight') * F.col('normalized_log_weight'))

# COMMAND ----------

# Update all data to only include LASSO features

lasso_indices = [0, 1, 3, 4, 6, 7, 10, 12, 13, 15, 17, 18, 19, 20, 21, 22, 24, 25, 27, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 45, 49, 52, 55, 58, 62, 63, 64, 65, 66, 67, 68]

slicer_lasso = VectorSlicer(inputCol="scaledFeatures", outputCol="Features", indices=lasso_indices)

test_df = test_df.filter(col('YEAR') == 2019)

train_lasso = slicer_lasso.transform(train_df)
test_lasso = slicer_lasso.transform(test_df)

# Uncomment this line if not including val_df in training cut
# val_lasso = slicer_lasso.transform(val_df)


# COMMAND ----------

# MAGIC %md *Run Final MLP Model*

# COMMAND ----------

def finalMLP(label, layers, train, val, test):
    trainingMetrics = []
    validationMetrics = []
    testMetrics = []

    # create the trainer and set its parameters
    trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234, labelCol=label, featuresCol = 'Features')

    evaluator_f1 = MulticlassClassificationEvaluator(labelCol=label, metricName="f1")
    evaluator_auc = BinaryClassificationEvaluator(labelCol=label, metricName="areaUnderROC")
    evaluator_precision = MulticlassClassificationEvaluator(labelCol=label, metricName="weightedPrecision")
    evaluator_recall = MulticlassClassificationEvaluator(labelCol=label, metricName="weightedRecall")

    # train the model
    model = trainer.fit(train)

    train_predictions = model.transform(train)
    trainingMetrics.append([evaluator_f1.evaluate(train_predictions), evaluator_auc.evaluate(train_predictions), evaluator_precision.evaluate(train_predictions), evaluator_recall.evaluate(train_predictions)])

    if val != None:
        # Make predictions on the validation data and evaluate
        val_predictions = model.transform(val)
        validationMetrics.append([evaluator_f1.evaluate(val_predictions), evaluator_auc.evaluate(val_predictions), evaluator_precision.evaluate(val_predictions), evaluator_recall.evaluate(val_predictions)])

    # Make predictions on the test data and evaluate            
    test_predictions = model.transform(test)
    testMetrics.append([evaluator_f1.evaluate(test_predictions), evaluator_auc.evaluate(test_predictions), evaluator_precision.evaluate(test_predictions), evaluator_recall.evaluate(test_predictions)])

    return (trainingMetrics, validationMetrics, testMetrics)

# COMMAND ----------

# This is for the version that doesn't use val_df in training cut
#use one layer MLP to do final predictions
layer_len = len(train_lasso.select("Features").first()[0]) #+ len(val_lasso.select("Features").first()[0]) + len(test_lasso.select("Features").first()[0])
layers = [layer_len, int((2/3 * layer_len)+1), 2]

print(layers)

# COMMAND ----------

# This version does not include val_df in the training cut

trainMetrics, validationMetrics, testMetrics = finalMLP('DEP_DEL15', layers, train_lasso, val_lasso, test_lasso)

print("MLP final Train Metrics:", trainMetrics)
print("MLP Validation Metrics:", validationMetrics)
print("MLP final Test Metrics:", testMetrics)

# COMMAND ----------

# This version does use val_df in the training cut
#use one layer MLP to do final predictions
layer_len = len(train_lasso.select("Features").first()[0]) 
layers = [layer_len, int((2/3 * layer_len)+1), 2]

print(layers)

trainMetrics, validationMetrics, testMetrics = finalMLP('DEP_DEL15', layers, train_lasso, None, test_lasso)

print("MLP Train Metrics:", trainMetrics)
print("MLP Validation Metrics:", validationMetrics)
print("MLP final test Metrics:", testMetrics)

# COMMAND ----------

# This version does use val_df in the training cut
# This version includes precision and recall

#use one layer MLP to do final predictions
layer_len = len(train_lasso.select("Features").first()[0]) 
layers = [layer_len, int((2/3 * layer_len)+1), 2]

print(layers)

trainMetrics, validationMetrics, testMetrics = finalMLP('DEP_DEL15', layers, train_lasso, None, test_lasso)

print("MLP Train Metrics:", trainMetrics)
print("MLP Validation Metrics:", validationMetrics)
print("MLP final test Metrics:", testMetrics)

# COMMAND ----------

def finalMLPGap(label, layers, train, val):
    trainingMetrics = []
    validationMetrics = []
    testMetrics = []

    # create the trainer and set its parameters
    trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234, labelCol=label, featuresCol = 'Features')

    evaluator_f1 = MulticlassClassificationEvaluator(labelCol=label, metricName="f1")
    evaluator_auc = BinaryClassificationEvaluator(labelCol=label, metricName="areaUnderROC")
    evaluator_precision = MulticlassClassificationEvaluator(labelCol=label, metricName="weightedPrecision")
    evaluator_recall = MulticlassClassificationEvaluator(labelCol=label, metricName="weightedRecall")

    # train the model
    model = trainer.fit(train)

    train_predictions = model.transform(train)
    trainingMetrics.append([evaluator_f1.evaluate(train_predictions), evaluator_auc.evaluate(train_predictions), evaluator_precision.evaluate(train_predictions), evaluator_recall.evaluate(train_predictions)])

    if val != None:
        # Make predictions on the validation data and evaluate
        val_predictions = model.transform(val)
        validationMetrics.append([evaluator_f1.evaluate(val_predictions), evaluator_auc.evaluate(val_predictions), evaluator_precision.evaluate(val_predictions), evaluator_recall.evaluate(val_predictions)])

    
    return (trainingMetrics, train_predictions, validationMetrics, val_predictions)

# COMMAND ----------

# This version does not include val_df in the training cut

layer_len = len(train_lasso.select("Features").first()[0]) 
layers = [layer_len, int((2/3 * layer_len)+1), 2]
trainMetrics, trainPred, validationMetrics, valPred = finalMLPGap('DEP_DEL15', layers, train_lasso, val_lasso)

print("MLP final Train Metrics:", trainMetrics)
print("MLP Validation Metrics:", validationMetrics)


# COMMAND ----------

print(trainPred.collect())

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Evaluation

# COMMAND ----------

# DELETE this chunk

mlp1_val_avg = {'Seed x': [[0.8150730756255622, 0.7245391912426341], [0.811343249636342, 0.7258747053382129], [0.8135492553631161, 0.7248227087449705], [0.8132163702882337, 0.7193826906466496], [0.8164552537850186, 0.7345434676494653]]}

# average_scores(mlp1_val_avg)

mlp1_train_avg = {'Seed x': [[0.7906628960914058, 0.7407576163284338], [0.8244596189167971, 0.736543401639633], [0.8034814106886093, 0.747757700164545], [0.8068852873020792, 0.7452809093390458], [0.8157468509052566, 0.7406628725395927]]}

# average_scores(mlp1_train_avg)

mlp2_val_avg = {'Seed x': [[0.8164982812695509, 0.7253652435384412], [0.8027405826730221, 0.728565695840542], [0.7543444690373181, 0.7274891168088744], [0.8154779729244402, 0.7227148865482198], [0.8170513808758793, 0.7356381581956242]]}

# average_scores(mlp2_val_avg)

mlp2_train_avg = {'Seed x': [[0.7918205886945304, 0.7392804866851961], [0.8160275834666753, 0.7367553386393969], [0.7309775271925639, 0.74565617705355], [0.8071237255755075, 0.7457577751587301], [0.8162132734528096, 0.7413455184877109]]}

average_scores(mlp2_train_avg)

# COMMAND ----------

# Compare training and validation F1 across all models

# DELETE include LSTM results? Not sure if we have training F1s

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Validation metrics (from big val)
init_noderived_f1_val = 0.604 # Initial logistic without derived features from phase ii
init_derived_f1_val = 0.422 # Initial logistic with all features from phase iii
lasso_log_f1_val = 0.743 # Logistic using 38 lasso features + 5 derived
mlp1_val = 0.814 # MLP with 1 hidden layer using lasso + derived
mlp2_val = 0.801 # MLP with 2 hidden layers using lasso + derived
rf_val = 0.756 # Random Forest using lasso + derived
gbdt_val = 0.791 # GBDT using lasso + derived
mlp_final_test = 0.803 # Final MLP w lasso + derived on full train instead of folds and test instead of big val

# Training metrics
init_noderived_f1_train = 0.648
init_derived_f1_train = 0.741
lasso_log_f1_train = 0.740
mlp1_train = 0.808
mlp2_train = 0.792
rf_train = 0.755
gbdt_train = 0.783
mlp_final_train = 0.805

# Bar positions
bar_positions = np.arange(8)  

# Bar heights
bar_heights_val = [init_noderived_f1_val, init_derived_f1_val, lasso_log_f1_val, mlp1_val, mlp2_val, rf_val, gbdt_val, mlp_final_test]
bar_heights_train = [init_noderived_f1_train, init_derived_f1_train, lasso_log_f1_train, mlp1_train, mlp2_train, rf_train, gbdt_train, mlp_final_train]

# Bar width
bar_width = 0.35

# Create a grouped bar chart
bars1 = plt.bar(bar_positions - bar_width/2, bar_heights_val, bar_width, label='Validation', color='blue')
bars2 = plt.bar(bar_positions + bar_width/2, bar_heights_train, bar_width, label='Training', color='orange')

# Set labels and title
plt.xlabel('Models')
plt.ylabel('Values')
plt.title('F1 Train and Validation Scores')

# Set x-axis ticks and labels
plt.xticks(bar_positions, ['Init Log', 'All Log', 'Lasso Log', 'MLP1', 'MLP2', 'RF', 'GBDT', 'Final MLP'])

# Add a legend
plt.legend(bbox_to_anchor=(1, 1))

# Add numerical values on top of the bars
# for bar1, bar2 in zip(bars1, bars2):
#     plt.text(bar1.get_x() + bar1.get_width() / 2 - 0.08, bar1.get_height() + 0.01, f'{bar1.get_height():.3f}', ha='center', va='bottom', color='black', fontweight='bold')
#     plt.text(bar2.get_x() + bar2.get_width() / 2 + 0.08, bar2.get_height() + 0.01, f'{bar2.get_height():.3f}', ha='center', va='bottom', color='black', fontweight='bold')

# Show the plot
plt.show()

# COMMAND ----------

# Compare training and validation F1 across all models except final MLP

# DELETE include LSTM results? Not sure if we have training F1s

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Validation metrics (from big val)
init_noderived_f1_val = 0.604 # Initial logistic without derived features from phase ii
init_derived_f1_val = 0.422 # Initial logistic with all features from phase iii
lasso_log_f1_val = 0.743 # Logistic using 38 lasso features + 5 derived
mlp1_val = 0.814 # MLP with 1 hidden layer using lasso + derived
mlp2_val = 0.801 # MLP with 2 hidden layers using lasso + derived
rf_val = 0.756 # Random Forest using lasso + derived
gbdt_val = 0.791 # GBDT using lasso + derived

# Training metrics
init_noderived_f1_train = 0.648
init_derived_f1_train = 0.741
lasso_log_f1_train = 0.740
mlp1_train = 0.808
mlp2_train = 0.792
rf_train = 0.755
gbdt_train = 0.783

# Bar positions
bar_positions = np.arange(7)  

# Bar heights
bar_heights_val = [init_noderived_f1_val, init_derived_f1_val, lasso_log_f1_val, mlp1_val, mlp2_val, rf_val, gbdt_val]
bar_heights_train = [init_noderived_f1_train, init_derived_f1_train, lasso_log_f1_train, mlp1_train, mlp2_train, rf_train, gbdt_train]

# Bar width
bar_width = 0.35

# Create a grouped bar chart
bars1 = plt.bar(bar_positions - bar_width/2, bar_heights_val, bar_width, label='Validation', color='blue')
bars2 = plt.bar(bar_positions + bar_width/2, bar_heights_train, bar_width, label='Training', color='orange')

# Set labels and title
plt.xlabel('Models')
plt.ylabel('Values')
plt.suptitle('F1 Train and Validation Scores')
plt.title('Avg Across Folds')

# Set x-axis ticks and labels
plt.xticks(bar_positions, ['Init Log', 'All Log', 'Lasso Log', 'MLP1', 'MLP2', 'RF', 'GBDT'])

# Add a legend
plt.legend(bbox_to_anchor=(1, 1))

# Add numerical values on top of the bars
# for bar1, bar2 in zip(bars1, bars2):
#     plt.text(bar1.get_x() + bar1.get_width() / 2 - 0.08, bar1.get_height() + 0.01, f'{bar1.get_height():.3f}', ha='center', va='bottom', color='black', fontweight='bold')
#     plt.text(bar2.get_x() + bar2.get_width() / 2 + 0.08, bar2.get_height() + 0.01, f'{bar2.get_height():.3f}', ha='center', va='bottom', color='black', fontweight='bold')

# Show the plot
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Archive

# COMMAND ----------

# Define final GBDT function

def trainGBDTModel_final(train_df, label, feature, test_df, seeds, maxIter=10, stepSize=0.1, featureSubsetStrategy='all'):
    trainingMetrics = {}
    testMetrics = {}

    # Run for multiple seeds    
    for seed in seeds:
        seed_name = f"Seed #{seed}" 
        trainingMetrics[seed_name] = []
        testMetrics[seed_name] = []

        # Create GBDT model instance
        gbt = GBTClassifier(labelCol=label, featuresCol=feature, weightCol='combined_weight', seed=seed, maxIter=maxIter, stepSize=stepSize, featureSubsetStrategy=featureSubsetStrategy)

        # Create a Pipeline
        pipeline = Pipeline(stages=[gbt])

        # Evaluation metrics
        # DELETE see if still works if move this outside the loop
        evaluator_f1 = MulticlassClassificationEvaluator(labelCol='DEP_DEL15', metricName="f1")
        evaluator_auc = BinaryClassificationEvaluator(labelCol='DEP_DEL15', metricName="areaUnderROC")

        # Train the model
        model = pipeline.fit(train_df)

        # Access feature importances
        feature_importances = model.stages[-1].featureImportances
        print(f"Seed #{seed} Feature Importances: {feature_importances}")

        # Make predictions on the training data and evaluate
        train_predictions = model.transform(train_df)
        trainingMetrics[seed_name].append([evaluator_f1.evaluate(train_predictions), evaluator_auc.evaluate(train_predictions)])

        # Make predictions on the validation data and evaluate
        test_predictions = model.transform(test_df)
        testMetrics[seed_name].append([evaluator_f1.evaluate(test_predictions), evaluator_auc.evaluate(test_predictions)])

    return (trainingMetrics, testMetrics)

# COMMAND ----------

# Run GBDT on 38 lasso + 5 derived features on final train and test cuts
# This version does use val_df in the training cut

seeds = [66]

gbdt_trainingMetrics_final, gbdt_testMetrics_final = trainGBDTModel_final(train_lasso, "DEP_DEL15", 'Features', test_lasso, seeds, 10, 0.1, "auto")

# COMMAND ----------

gbdt_trainingMetrics_final

# COMMAND ----------

gbdt_testMetrics_final

# COMMAND ----------

# Define final RF function

def trainRandomForestModel_final(train_df, label, feature, test_df, numTrees, seeds, featureSubsetStrategy, maxDepth):
    trainingMetrics = {}
    testMetrics = {}

    # Run for multiple seeds    
    for seed in seeds:
        seed_name = f"Seed #{seed}" 
        trainingMetrics[seed_name] = []
        testMetrics[seed_name] = []

        # Create a Random Forest model instance
        rf = RandomForestClassifier(labelCol=label, featuresCol=feature, numTrees=numTrees, weightCol='combined_weight', seed=seed, featureSubsetStrategy=featureSubsetStrategy, maxDepth=maxDepth)

        # Evaluation metrics
        evaluator_f1 = MulticlassClassificationEvaluator(labelCol=label, metricName="f1")
        evaluator_auc = BinaryClassificationEvaluator(labelCol=label, metricName="areaUnderROC")

        # Fit model
        model = rf.fit(train_df)

        # Access feature importances
        feature_importances = model.featureImportances
        print(f"Seed #{seed} Feature Importances: {feature_importances}")

        # Make predictions on the training data and evaluate
        train_predictions = model.transform(train_df)
        trainingMetrics[seed_name].append([evaluator_f1.evaluate(train_predictions), evaluator_auc.evaluate(train_predictions)])

        # Make predictions on the validation data and evaluate
        test_predictions = model.transform(test_df)
        testMetrics[seed_name].append([evaluator_f1.evaluate(test_predictions), evaluator_auc.evaluate(test_predictions)])
    
    return (trainingMetrics, testMetrics)

# COMMAND ----------

# Run Random Forest on 38 lasso + 5 derived features on all folds
# This version does use val_df in the training cut

seeds = [78]

rf_trainingMetrics_final, rf_testMetrics_final = trainRandomForestModel_final(train_lasso, "DEP_DEL15", 'Features', test_lasso, 10, seeds, "auto", 5)

# COMMAND ----------

rf_trainingMetrics_final

# COMMAND ----------

rf_testMetrics_final

# COMMAND ----------

