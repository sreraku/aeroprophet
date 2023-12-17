# Databricks notebook source
# MAGIC %md
# MAGIC # Imports

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql.functions import col,isnan, when, count, col, split, trim, lit, avg, sum, expr
from pyspark.sql import functions as F
import holidays
import seaborn as sns

# COMMAND ----------

# MAGIC %md
# MAGIC # Connect to Storage

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
# MAGIC ## Resources & Notes
# MAGIC #### Delete this chunk before submission
# MAGIC - Description of what should be in each section below: https://digitalcampus.instructure.com/courses/14487/pages/phase-descriptions-and-deliverables?module_item_id=1711798
# MAGIC - This is the main notebook
# MAGIC   - There should be no code in this notebook
# MAGIC   - Include links in this notebook to separate coding notebooks
# MAGIC - Clone w261_final_project_starter_nb_fp BUT DON'T MODIFY IT DIRECTLY
# MAGIC   - https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/365886751420832/command/365886751420833
# MAGIC   - The starter notebook has everything you need to access the data. All data is accessible via Databricks and is stored on the Azure Cloud.
# MAGIC - Flights data dictionary: https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ
# MAGIC - Flight data
# MAGIC   - dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/  # All flights during 2015 to 2021 inclusive
# MAGIC   - dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_3m/   #2015 Q1 
# MAGIC   - dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_6m/   #2015 Q1 and Q2
# MAGIC   - dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_1y/    # 2019 (correct 2019!)
# MAGIC - Weather data dictionary p.8-12: chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf
# MAGIC - Weather data
# MAGIC   - dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data/          #weather data for 2015-2021 
# MAGIC   - dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data_1y/    #weather data for all of 2019
# MAGIC   - dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data_3m/   #weather data for 2015 Q1
# MAGIC   - dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data_6m/   #weather data for 2015 Q1 and Q2
# MAGIC - Airport data
# MAGIC   - dbfs:/mnt/mids-w261/datasets_final_project_2022/stations_data
# MAGIC - Airport code conversions
# MAGIC   - https://datahub.io/core/airport-codes
# MAGIC - We have rejoined joined Arrival Time Performance with local Weather  data. This produces the ATPW dataset. Please see Azure storage bucket via the Databricks starter notebook on how to access this data:
# MAGIC   - 3 months ATPW
# MAGIC   - 1 year ATPW
# MAGIC   - 3-5 years ATPW

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Data

# COMMAND ----------

# MAGIC %md
# MAGIC Read in Our Joined Data

# COMMAND ----------

pd.set_option('display.max_rows', None)

# COMMAND ----------

# MAGIC %md
# MAGIC Convert each field to the proper datatypes

# COMMAND ----------

# MAGIC %md
# MAGIC Review Distribution

# COMMAND ----------

# MAGIC %md
# MAGIC Read in Transformed Parquet

# COMMAND ----------

# MAGIC %md
# MAGIC Create K Cross Folds

# COMMAND ----------

from collections import namedtuple
from datetime import date, timedelta
import re

BlockSplit = namedtuple("BlockSplit", ["min_date", "train_cut", "val_cut"])

def make_block_splits(min_date, max_date, train_width, val_width):
    blocks = list()
    train_min = min_date
    val_cut = min_date
    
    # Loop over and create blocks
    while True:
        train_cut = train_min + train_width
        val_cut = train_cut + val_width
        if train_cut > max_date or val_cut > max_date:
            break
        blocks.append(BlockSplit(train_min, train_cut, val_cut))
        train_min = val_cut
    
    return blocks

def make_folds(df, date_col, splits):
    """
    Make folds using the specified block splits.

    Args:
        df - the dataframe to make folds from
        date_col - the name of a date column to split on
        splits - a list of ``BlockSplit`` instances
    
    Returns:
        a list of (train_df, val_df) tuples
    """
    folds = list()
    for split in splits:
        train_df = df.filter((df[date_col] >= split.min_date) & (df[date_col] < split.train_cut))
        val_df = df.filter((df[date_col] >= split.train_cut) & (df[date_col] < split.val_cut))
        folds.append((train_df, val_df))
    return folds

def save_folds_to_blob(folds, blob_url, fold_name):
    for i, (train_df, val_df) in enumerate(folds):
        dbutils.fs.rm(f"{blob_url}/{fold_name}/train_{i}_df", recurse=True)
        dbutils.fs.rm(f"{blob_url}/{fold_name}/val_{i}_df", recurse=True)
        train_df.write.parquet(f"{blob_url}/{fold_name}/train_{i}_df")
        val_df.write.parquet(f"{blob_url}/{fold_name}/val_{i}_df")

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


# COMMAND ----------

folds = load_folds_from_blob_and_cache(team_blob_url, 'k_folds_cross_val')

# COMMAND ----------

# MAGIC %md
# MAGIC Train Models

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

trainingMetrics = []
validationMetrics = []

# Create a Logistic Regression model instance
lr = LogisticRegression(labelCol="DEP_DEL15", featuresCol="scaledFeatures")

# Evaluator for binary classification
evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15")

for fold in folds:
    train_df, val_df = fold

    # Fit the model on the training data
    model = lr.fit(train_df)

    # Make predictions on the training data and evaluate
    train_predictions = model.transform(train_df)
    train_accuracy = evaluator.evaluate(train_predictions)
    trainingMetrics.append(train_accuracy)

    # Make predictions on the validation data and evaluate
    val_predictions = model.transform(val_df)
    val_accuracy = evaluator.evaluate(val_predictions)
    validationMetrics.append(val_accuracy)

print("Training Metrics:", trainingMetrics)
print("Validation Metrics:", validationMetrics)


# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Assuming you have your feature and label columns defined as 'features' and 'label'
feature_col = 'scaledFeatures'
label_col = 'DEP_DEL15'

# Create a Linear Regression model with Lasso regularization
lasso_model = LogisticRegression()

# Define the hyperparameter grid for Lasso regression
param_grid = ParamGridBuilder() \
              .addGrid(lasso_model.labelCol, ['DEP_DEL15']) \
              .addGrid(lasso_model.featuresCol, 'scaledFeatures') \
              .addGrid(lasso_model.weightCol, ['weight']) \
              .addGrid(lasso_model.elasticNetParam, [1]) \
               .addGrid(lasso_model.regParam, [0.01, 0.02, 0.03]) \
               .addGrid(lasso_model.maxIter, [10, 20, 30]) \
               .build()

# Initialize an empty list to store the results
results = []

# Iterate through each set of hyperparameters
for params in param_grid:
    fold_results = []
    for fold in folds:
        train_df, val_df = fold
        # Set the hyperparameters in the pipeline
        param = {str(key): value for key, value in params.items()}
        lasso_model.setParams(**param)
        
        # Create a pipeline with the Lasso model
        pipeline = Pipeline(stages=[lasso_model])
        
        # Fit the model
        model = pipeline.fit(train_df)
        
        # Make predictions on the validation set
        predictions = model.transform(val_df)
        
        # Evaluate the model using the specified evaluator
        evaluator_f1 = MulticlassClassificationEvaluator(labelCol="DEP_DEL15", metricName="f1")
        metric = evaluator_f1.evaluate(predictions)
        
        # Store the results
        fold_results.append(metric)
    
    results.append((params,avg(fold_results)))

# Find the best set of hyperparameters
best_params, best_metric = max(results, key=lambda x: x[1])

print(best_params, best_metric)
    

# COMMAND ----------

param_grid = ParamGridBuilder() \
              .addGrid(lasso_model.labelCol, ['DEP_DEL15']) \
              .addGrid(lasso_model.featuresCol, 'scaledFeatures') \
              .addGrid(lasso_model.weightCol, ['weight']) \
              .addGrid(lasso_model.elasticNetParam, [1]) \
               .addGrid(lasso_model.regParam, [0.01, 0.02, 0.03]) \
               .addGrid(lasso_model.maxIter, [10, 20, 30]) \
               .build()

# COMMAND ----------

for param in param_grid:
    params = {str(key): value for key, value in param.items()}
    print(params)

# COMMAND ----------

! pip install hyperopt

# COMMAND ----------

from hyperopt import hp, Trials, fmin, tpe, STATUS_OK
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import time


search_space = {"regParam": hp.quniform("regParam",0.01, 0.1,1)}

def hyperparameter_tuning_lr(folds):
    num_evals = 5
    trials = Trials()
    best_hyperparam = fmin(fn=objective_function, space=search_space, algo=tpe.suggest, max_evals=num_evals,trials=trials,rstate=np.random.default_rng(1))
    return(best_hyperparam)

def objective_function(params):
    
    regParam = params["regParam"]
    
    def train_baseline_folds(folds):
        """
        Get validation  f1 score across all folds

        Parameters:
            folds: sets of train and validation datasets
        
        returns:
            average validation f1 score
        """
        val_results = pd.DataFrame()
        for i, (train_df, val_df) in enumerate(folds):
            print(f"Training fold {i}: ", end="")
            start = time.time()
            results = train_baseline(train_df, val_df)
            elapsed = timedelta(seconds=time.time() - start)
            print(f"DONE. F1score:{float(results):0.3f}(elapsed: {elapsed})")
            val_results = pd.concat([val_results, pd.DataFrame([results])], ignore_index=True)
        return(val_results.mean())
    
    def train_baseline(train,val):
        """
        Parameters:
            test_df: dataframe holding the test set

        returns:
            balanced f1 score
        """
        # Use logistic regression 
        lr = LogisticRegression(regParam=regParam,labelCol='DEP_DEL15', weightCol='weight',featuresCol = "scaledFeatures", elasticNetParam = 1)
        evaluatorF1 = MulticlassClassificationEvaluator(labelCol="DEP_DEL15", metricName="f1")

        # Build our ML pipeline
        pipeline = Pipeline(stages=[lr])

        model = pipeline.fit(train)
        val_pred = model.transform(val)
        f1score = evaluatorF1.evaluate(val_pred)
        return f1score
    loss = -train_baseline_folds(folds)
    return{'loss': loss, 'status': STATUS_OK}

hyperparameter_tuning_lr(folds)


# COMMAND ----------

hyperparameter_tuning_lr(folds)

# COMMAND ----------

def calcScores(feat_list):
    for validation_metrics in feat_list:
        # Transpose the matrix to get lists of metrics
        print (validation_metrics)
        
        f1_score, weighted_precision, weighted_recall, AuC = zip(*validation_metrics)

        # Calculate averages
        average_f1_score = sum(f1_score) / len(f1_score)
        average_w_precision = sum(weighted_precission) / len(weighted_precission)
        average_w_recall = sum(weighted_recall) / len(weighted_recall)
        average_AuC = sum(AuC) / len(AuC)
        # f1, wprec, wrecall, auc
        # Print the results
        print("Average F1 Score:", average_f1_score)
        print("Average Weighted precision:", average_w_precision)
        print("Average Weighted recall:", average_w_recall)
        print("Average AuC:", average_AuC)
        
feat_lst = [
    [0.6443993149249017, 0.8078082684942527, 0.5886079421332077, 0.7117815140510514],
    [0.7141702626497349, 0.7844903944880464, 0.684824458559949, 0.7442511108127259],
    [0.666058311063887, 0.7873562558443744, 0.6247148293685167, 0.7317664020864817],
    [0.6082090520897839, 0.7697697767277337, 0.5598282391578777, 0.6855099973919342],
    [0.7943994133679596, 0.805052748318414, 0.785989724849017, 0.7208277362195241]
]
calcScores([feat_lst])

# COMMAND ----------

feat1 = [
    [0.5932808284784616, 0.7912381275762586, 0.5325401497169415, 0.6378139450879167], [0.7000630352560887, 0.7480308481456186, 0.6736016811772045, 0.6674149881902408], [0.6516102466675864, 0.7543797555642846, 0.6090199630801361, 0.64169820316601], [0.5966345217523275, 0.7524162437017295, 0.5465548552040593, 0.6207089047102026], [0.574932671915352, 0.7799485868167875, 0.5181203676119494, 0.6345823970186266]
    ]

feat2 = [
    [0.6030111197166523, 0.7919650834767249, 0.5427754631043564, 0.6412081642429228], [0.6931070336201022, 0.7479893614279898, 0.6643614689531565, 0.6659850796846776], [0.6441981694802096, 0.7565222834802887, 0.6002889811678953, 0.6448521892941288], [0.6420446687795556, 0.7460803745810989, 0.5971348795818955, 0.6159047922708888], [0.591877965886093, 0.7797216563694561, 0.5353106099063076, 0.6387512030829677]
    ]

feat3 = [
    [0.6099875656596918, 0.7899848029648574, 0.5500360461374998, 0.6371757653873165], [0.7112288016982599, 0.7439136423355263, 0.690458241280003, 0.6645741400026671], [0.6565465760071816, 0.7551717297199465, 0.6148223050833175, 0.6442028178573888], [0.7141223364205527, 0.7303859497545047, 0.7007783660472686, 0.5880064802038381], [0.6517936756190951, 0.7730022293726455, 0.6010847881138931, 0.637534371167174]
    ]

feat4 = [
    [0.5215314477066739, 0.7938772481497248, 0.4632276874797933, 0.6281083539950946], [0.6681012431173384, 0.7476891345416085, 0.6327795468399846, 0.6610743297535637], [0.6102732482579019, 0.7587703293960278, 0.5627226685403353, 0.641746927549784], [0.4433858959233231, 0.7274000252158188, 0.40307672351446655, 0.571634177833108], [0.6639449689395649, 0.770229026771512, 0.6157971547561134, 0.6333340031551813]
    ]

feat5 = [
    [0.5667980088871306, 0.7921785954571748, 0.5058671751383195, 0.6345189792088447], [0.6634235248225755, 0.7474059148969602, 0.6271334771307423, 0.6587819074943464], [0.6228734047647655, 0.7564832485424837, 0.5762807429740524, 0.6402951885092412], [0.7058625074811239, 0.7319118875870507, 0.6864096462025215, 0.5927344575744828], [0.6518922453736913, 0.7718654873214628, 0.6012387133517637, 0.6351328422735044]
    ]
feat_list = [feat1, feat2, feat3, feat4, feat5]
calcScores(feat_lst)

# COMMAND ----------

# calculate averages to identify which feature combinations result in strongest baseline performance:
feat1 = [
    [0.5932808284784616, 0.7912381275762586, 0.5325401497169415, 0.6378139450879167], [0.7000630352560887, 0.7480308481456186, 0.6736016811772045, 0.6674149881902408], [0.6516102466675864, 0.7543797555642846, 0.6090199630801361, 0.64169820316601], [0.5966345217523275, 0.7524162437017295, 0.5465548552040593, 0.6207089047102026], [0.574932671915352, 0.7799485868167875, 0.5181203676119494, 0.6345823970186266]
    ]

feat2 = [
    [0.6030111197166523, 0.7919650834767249, 0.5427754631043564, 0.6412081642429228], [0.6931070336201022, 0.7479893614279898, 0.6643614689531565, 0.6659850796846776], [0.6441981694802096, 0.7565222834802887, 0.6002889811678953, 0.6448521892941288], [0.6420446687795556, 0.7460803745810989, 0.5971348795818955, 0.6159047922708888], [0.591877965886093, 0.7797216563694561, 0.5353106099063076, 0.6387512030829677]
    ]

feat3 = [
    [0.6099875656596918, 0.7899848029648574, 0.5500360461374998, 0.6371757653873165], [0.7112288016982599, 0.7439136423355263, 0.690458241280003, 0.6645741400026671], [0.6565465760071816, 0.7551717297199465, 0.6148223050833175, 0.6442028178573888], [0.7141223364205527, 0.7303859497545047, 0.7007783660472686, 0.5880064802038381], [0.6517936756190951, 0.7730022293726455, 0.6010847881138931, 0.637534371167174]
    ]

feat4 = [
    [0.5215314477066739, 0.7938772481497248, 0.4632276874797933, 0.6281083539950946], [0.6681012431173384, 0.7476891345416085, 0.6327795468399846, 0.6610743297535637], [0.6102732482579019, 0.7587703293960278, 0.5627226685403353, 0.641746927549784], [0.4433858959233231, 0.7274000252158188, 0.40307672351446655, 0.571634177833108], [0.6639449689395649, 0.770229026771512, 0.6157971547561134, 0.6333340031551813]
    ]

feat5 = [
    [0.5667980088871306, 0.7921785954571748, 0.5058671751383195, 0.6345189792088447], [0.6634235248225755, 0.7474059148969602, 0.6271334771307423, 0.6587819074943464], [0.6228734047647655, 0.7564832485424837, 0.5762807429740524, 0.6402951885092412], [0.7058625074811239, 0.7319118875870507, 0.6864096462025215, 0.5927344575744828], [0.6518922453736913, 0.7718654873214628, 0.6012387133517637, 0.6351328422735044]
    ]
feat_list = [feat1, feat2, feat3, feat4, feat5]

for validation_metrics in feat_list:
    # Transpose the matrix to get lists of metrics
    f1_score, weighted_precission, weighted_recall, AuC = zip(*validation_metrics)

    # Calculate averages
    average_f1_score = sum(f1_score) / len(f1_score)
    average_w_precision = sum(weighted_precission) / len(weighted_precission)
    average_w_recall = sum(weighted_recall) / len(weighted_recall)
    average_AuC = sum(AuC) / len(AuC)
# f1, wprec, wrecall, auc
    # Print the results
    print("Average F1 Score:", average_f1_score)
    print("Average Weighted precision:", average_w_precision)
    print("Average Weighted recall:", average_w_recall)
    print("Average AuC:", average_AuC)

# COMMAND ----------

