# Databricks notebook source
# MAGIC %md
# MAGIC ### Imports

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from collections import namedtuple
from datetime import date, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import re

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
# MAGIC ### Load Data

# COMMAND ----------

df_test = spark.read.parquet(f"{team_blob_url}/test_transformed_2023_12_10").cache()
df_val = spark.read.parquet(f"{team_blob_url}/val_transformed_2023_12_10").cache()

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

# MAGIC %md
# MAGIC ### Initial Baseline Models

# COMMAND ----------

# MAGIC %md Define Logistic Regression Function

# COMMAND ----------

# Define Logistic Regression Function

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def trainLogRegModel(folds, label, feature, final_val):
    trainingMetrics = []
    validationMetrics = []
    finalValidationMetrics = []

    # Create a Logistic Regression model instance
    lr = LogisticRegression(labelCol=label, featuresCol=feature, weightCol = 'weight')

    # Evaluator for multiclass classification
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol=label, metricName="f1")
    evaluator_precision = MulticlassClassificationEvaluator(labelCol=label, metricName="weightedPrecision")
    evaluator_recall = MulticlassClassificationEvaluator(labelCol=label, metricName="weightedRecall")
    evaluator_auc = BinaryClassificationEvaluator(labelCol=label, metricName="areaUnderROC")

    for fold in folds:
        train_df, val_df = fold
        
        # Fit the model on the training data
        model = lr.fit(train_df)
        # Print model coeficients
        print(model.coefficients)

        # Make predictions on the training data and evaluate
        train_predictions = model.transform(train_df)
        trainingMetrics.append([evaluator_f1.evaluate(train_predictions), evaluator_precision.evaluate(train_predictions), evaluator_recall.evaluate(train_predictions), evaluator_auc.evaluate(train_predictions)])

        # Make predictions on the validation data and evaluate
        val_predictions = model.transform(val_df)
        validationMetrics.append([evaluator_f1.evaluate(val_predictions), evaluator_precision.evaluate(val_predictions), evaluator_recall.evaluate(val_predictions), evaluator_auc.evaluate(val_predictions)])
    if (final_val != None):
        final_val_predictions = model.transform(val_df)
        finalValidationMetrics.append([evaluator_f1.evaluate(final_val_predictions), evaluator_precision.evaluate(final_val_predictions), evaluator_recall.evaluate(final_val_predictions), evaluator_auc.evaluate(final_val_predictions)])
    return (trainingMetrics, validationMetrics, finalValidationMetrics)

# COMMAND ----------

# MAGIC %md Define Function to Create Folds with Different Feature Sets

# COMMAND ----------

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

    #features from feature set 3 from LASSO fold 3 coef
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

# MAGIC %md Run initial logistic models

# COMMAND ----------

# Folds with all 50 original features without derived features

# 11/29 run took 52.80 minutes

folds_noderived = final_feature_finder(folds, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,50,51])

# running logistic regression on original 50 features without derived features
trainingMetrics_init_noderived, validationMetrics_init_noderived, finalValidationMetrics = trainLogRegModel(folds_noderived, "DEP_DEL15", 'Features',df_val)

print("Logistical Regression Training Metrics:", trainingMetrics_init_noderived)
print("Logistical regression Validation Metrics:", validationMetrics_init_noderived)
print("Logistical regression Final Validation Metrics:", finalValidationMetrics)

# COMMAND ----------

# Run Logistic Regression on Initial Subset Data including derived features

# 11/28 run took 10.33 minutes

trainingMetrics_initlog0, validationMetrics_initlog0, _ = trainLogRegModel(fold0, "DEP_DEL15", "scaledFeatures", None)
# print(coefficients_list)
print("Initial Logistic Regression Training Metrics:", trainingMetrics_initlog0)
print("Initial Logistic Regression  Validation Metrics:", validationMetrics_initlog0)

# COMMAND ----------

# Run Logistic Regression on Initial Full Data including 4 derived features

# 11/28 run took 50.41 minutes 

trainingMetrics_initlogfull, validationMetrics_initlogfull, finalValidationMetrics = trainLogRegModel(folds, "DEP_DEL15", "scaledFeatures", df_val)
# print(coefficients_list)
print("Initial Logistic Regression Training Metrics:", trainingMetrics_initlogfull)
print("Initial Logistic Regression  Validation Metrics:", validationMetrics_initlogfull)
print("Initial Logistic Regression  final Validation Metrics:", finalValidationMetrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lasso Feature Selection

# COMMAND ----------

# MAGIC %md Define Function for LASSO Regression

# COMMAND ----------

#lasso regression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def trainLassoRegModel(folds, label, feature, final_val):
    trainingMetrics = []
    validationMetrics = []
    finalValidationMetrics = []
    coef = []

    # Create a Logistic Regression model instance
    lr = LogisticRegression(labelCol=label, featuresCol="scaledFeatures", weightCol = 'weight', regParam  = 0.01, elasticNetParam = 1.0)

    # Evaluator for multiclass classification
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol=label, metricName="f1")
    evaluator_precision = MulticlassClassificationEvaluator(labelCol=label, metricName="weightedPrecision")
    evaluator_recall = MulticlassClassificationEvaluator(labelCol=label, metricName="weightedRecall")
    # Hailee: Added this so we can do AUC plots. Can't do areaUnderROC with MulticlassClassificationEvaluator and BinaryClassificationEvaluator doesn't have weightCol arg
    evaluator_auc = BinaryClassificationEvaluator(labelCol=label, metricName="areaUnderROC")

    for fold in folds:
        train_df, val_df = fold

        # Fit the model on the training data
        model = lr.fit(train_df)

        # Zip feature names with coefficients
        # NOTE it would be helpful to save and return these in future runs so we can easily identify dropped features
        print(model.coefficients)
        coef.append(model.coefficients.toArray())

        # feature_importance = list(zip(col_names, model.coefficients))

        # print("Feature Importance:")
        # for feature, importance in feature_importance:
        #     print("  {}: {:.5f}".format(feature, importance))

        # Make predictions on the training data and evaluate
        train_predictions = model.transform(train_df)
        trainingMetrics.append([evaluator_f1.evaluate(train_predictions), evaluator_precision.evaluate(train_predictions), evaluator_recall.evaluate(train_predictions), evaluator_auc.evaluate(train_predictions)])

        # Make predictions on the validation data and evaluate
        val_predictions = model.transform(val_df)
        validationMetrics.append([evaluator_f1.evaluate(val_predictions), evaluator_precision.evaluate(val_predictions), evaluator_recall.evaluate(val_predictions), evaluator_auc.evaluate(val_predictions)])

    if(final_val != None):
        final_val_predictions = model.transform(final_val)
        finalValidationMetrics.append([evaluator_f1.evaluate(final_val_predictions), evaluator_precision.evaluate(final_val_predictions), evaluator_recall.evaluate(final_val_predictions), evaluator_auc.evaluate(final_val_predictions)])
    
    return (trainingMetrics, validationMetrics, finalValidationMetrics)

# COMMAND ----------

# MAGIC %md Train LASSO Models

# COMMAND ----------

# Run on subset data
# 11/27 run took 9.99 minutes and returns 22 variables

trainingMetrics0, validationMetrics0, _ = trainLassoRegModel(fold0, "DEP_DEL15", "scaledFeatures", None)
print("Lasso regression Training Metrics:", trainingMetrics0)
print("Lasso regression Validation Metrics:", validationMetrics0)

# COMMAND ----------

# Run on all folds
# All features including derived

# 11/27 runs took 41.80 minutes, 49.35 minutes

# NOTE If this gets run again, change names to trainingMetrics_lasso and validationMetrics_lasso
trainingMetrics, validationMetrics, finalValMetrics = trainLassoRegModel(folds, "DEP_DEL15", "scaledFeatures", df_val)
# print(coefficients_list)
print("Lasso regression Training Metrics:", trainingMetrics)
print("Lasso regression Validation Metrics:", validationMetrics)
print("Lasso regression Final Validation Metrics:", finalValMetrics)

# COMMAND ----------

# MAGIC %md
# MAGIC Drop features consistently driven to zero by lasso

# COMMAND ----------

from pyspark.ml.feature import VectorSlicer

feat_idx_map = pd.DataFrame(folds[0][0].schema["features"].metadata["ml_attr"]["attrs"]["binary"]+folds[0][0].schema["features"].metadata["ml_attr"]["attrs"]["numeric"]).sort_values("idx")

#final feature indexes
feature_idxs = list(set([0,1,2,6,8,12,15,22,23,25,26,32,37,39,40,41,43,46,48,49,50,51,52,53]))

# feature_idxs = list(set([0,1,2,4,7,8,11,12,18,21,28,32,36,40,41,43,46,50,51, 
#                     0,1,6,8,11,12,32,37,40,41,43,46,50,51, 
#                     0,1,2,6,8,12,15,22,23,25,26,32,37,39,40,41,43,46,50,51, 
#                     0,1,6,12,19,21,30,31,32,34,37,38,40,43,46,50,51,
#                     0,1,3,6,12,15,32,34,36,37,38,40,43,46,50,51]))

slicer = VectorSlicer(inputCol="scaledFeatures", outputCol="Lasso_Selected_Features", indices=feature_idxs)

new_folds = []
for fold in folds:
    train = slicer.transform(fold[0])
    val = slicer.transform(fold[1])

    new_folds.append((train, val))

print('Num of final features:', len(feature_idxs))
feat_idx_map.iloc[feature_idxs]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Grid Search

# COMMAND ----------

final_folds = final_feature_finder(folds, [0,1,2,6,8,12,15,22,23,25,26,32,37,39,40,41,43,46,48,49,50,51,52,53])

# COMMAND ----------

from hyperopt import hp, Trials, fmin, tpe, STATUS_OK
import numpy as np
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import time


search_space = {"regParam": hp.quniform("regParam",0.0, 0.1,0.01),
                "maxIter": hp.quniform("maxIter", 10,30,10)
                }

def hyperparameter_tuning_lr(folds):
    num_evals = 5
    trials = Trials()
    best_hyperparam = fmin(fn=objective_function, space=search_space, algo=tpe.suggest, max_evals=num_evals,trials=trials,rstate=np.random.default_rng(1))
    return(best_hyperparam)

def objective_function(params):
    
    regParam = params["regParam"]
    maxIter = params["maxIter"]
    
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
            balanced accuracy score
        """
        # Use logistic regression 
        lr = LogisticRegression(regParam=regParam,labelCol='DEP_DEL15', weightCol='weight',featuresCol = "scaledFeatures", elasticNetParam = 0, maxIter = maxIter)
        evaluatorF1 = MulticlassClassificationEvaluator(labelCol="DEP_DEL15", metricName="f1")

        # Build our ML pipeline
        pipeline = Pipeline(stages=[lr])

        model = pipeline.fit(train)
        val_pred = model.transform(val)
        f1score = evaluatorF1.evaluate(val_pred)
        return f1score
    loss = -train_baseline_folds(folds)
    return{'loss': loss, 'status': STATUS_OK}

hyperparameter_tuning_lr(final_folds)

# COMMAND ----------

# MAGIC %md Using the LASSO models

# COMMAND ----------

# Function to create columns for each feature set selected by the different folds in LASSO

from pyspark.ml.feature import VectorSlicer

def feature_finder(folds):
    """Parameters:
    - folds (list): A list of tuples, where each tuple represents a fold containing
                   a training set and a validation set.

    Returns:
    list: A new list of folds, where each fold includes the original training and
          validation sets augmented with additional features based on hard coded LASSO coefficients.
"""
    #not including derived columns
    #LASSO fold 1 coef
    slicer1 = VectorSlicer(inputCol="scaledFeatures", outputCol="LF1", indices=[0,1,2,4,7,8,11,12,18,21,28,32,36,40,41,43,46,50,51])
    #LASSO fold 2 coef
    slicer2 = VectorSlicer(inputCol="scaledFeatures", outputCol="LF2", indices=[0,1,6,8,11,12,32,37,40,41,43,46,50,51])
    #LASSO fold 3 coef
    slicer3 = VectorSlicer(inputCol="scaledFeatures", outputCol="LF3", indices=[0,1,2,6,8,12,15,22,23,25,26,32,37,39,40,41,43,46,50,51])
    #LASSO fold 4 coef
    slicer4 = VectorSlicer(inputCol="scaledFeatures", outputCol="LF4", indices=[0,1,6,12,19,21,30,31,32,34,37,38,40,43,46,50,51])
    #LASSO fold 5 coef
    slicer5 = VectorSlicer(inputCol="scaledFeatures", outputCol="LF5", indices=[0,1,3,6,12,15,32,34,36,37,38,40,43,46,50,51])

    def feature_create(df):
    
        output = slicer1.transform(df)
        output = slicer2.transform(output)
        output = slicer3.transform(output)
        output = slicer4.transform(output)
        output = slicer5.transform(output)
        return output


    
    new_folds = []
    for train,val in folds:
        new_train = feature_create(train)
        new_val = feature_create(val)
        new_folds.append((new_train,new_val))
    return new_folds


lasso_folds = feature_finder(folds)


# COMMAND ----------

lasso_folds

# COMMAND ----------

#landon to set up different cols with subsets of feature vector assembler col from LASSO results, and EXCLUDING derived columns 48, 49, 52, 53
#feature col names: LF1, LF2, LF3, LF4, LF5
features = ["LF1", "LF2", "LF3", "LF4", "LF5"]
#Sreeram to manually run 5 iterations of baseline model, each with a different set of columns.
for feat in features:
    trainingMetrics, validationMetrics, finalValidationMetrics = trainLogRegModel(lasso_folds, "DEP_DEL15", feat, df_val)

    print("Logistical regression Training Metrics:", trainingMetrics)
    print("Logistical regression Validation Metrics:", validationMetrics)
    print("Logistical regression Validation Metrics:", finalValidationMetrics)
#compare model performance for ultimate selection of features to include.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Selection Using Different Feature Sets

# COMMAND ----------

# MAGIC %md Logistic Regressions on Different Feature Sets

# COMMAND ----------

# Logistic regression on 33 features which had values from LASSO. Not including derived features
trainingMetrics, validationMetrics, validationMetrics = trainLogRegModel(new_folds, "DEP_DEL15", 'Lasso_Selected_Features', df_val)
print("Logistical regression Training Metrics:", trainingMetrics)
print("Logistical regression Validation Metrics:", validationMetrics)
print("Logistical regression Validation Metrics:", finalValidationMetrics)

# COMMAND ----------

# Folds with the 20 lasso features and 4 derived features
final_folds = final_feature_finder(folds, [0,1,2,6,8,12,15,22,23,25,26,32,37,39,40,41,43,46,48,49,50,51,52,53])

# running logistic regression on selected 20 features + 4 dervied cols
final_trainingMetrics, final_validationMetrics, finalValidationMetrics = trainLogRegModel(final_folds, "DEP_DEL15", 'Features', df_val)

print("Logistical regression Training Metrics:", final_trainingMetrics)
print("Logistical regression Validation Metrics:", final_validationMetrics)
print("Logistical regression Validation Metrics:", finalValidationMetrics)

# COMMAND ----------

# MAGIC %md Calculate average performance across folds to see which feature set performed best

# COMMAND ----------

# Function to calculate the average scores of a feature set across folds

def calcScores(feat_list):
    metric_index = 0
    for validation_metrics in feat_list:
        # Transpose the matrix to get lists of metrics
        f1_score, weighted_precission, weighted_recall, AuC = zip(*validation_metrics)
        #store scores for future use
        f1_score_dict = {}
        # Calculate averages
        average_f1_score = sum(f1_score) / len(f1_score)
        average_w_precision = sum(weighted_precission) / len(weighted_precission)
        average_w_recall = sum(weighted_recall) / len(weighted_recall)
        average_AuC = sum(AuC) / len(AuC)
        # f1, wprec, wrecall, auc
        # Print the results
        print('Feature index: ',metric_index)
        print("Average F1 Score:", average_f1_score)
        print("Average Weighted precision:", average_w_precision)
        print("Average Weighted recall:", average_w_recall)
        print("Average AuC:", average_AuC)
        print('-'*30)
        metric_index +=1


# COMMAND ----------

# Calculate averages of validation metrics to identify which feature combinations result in strongest baseline performance:

# From those tested in previous section, Fold 3 Lasso features had strongest overall performance

# Results from the 20+4 regression (20 = feat3 = best performing lasso feature set)
final_feat = [
    [0.7042388499438903, 0.8078836653125334, 0.6589284512377267, 0.715430899134182], [0.7703352643231202, 0.7869560513054601, 0.7591492488686061, 0.7459307976697129], [0.6771763535296491, 0.7869340744977683, 0.6374657722955621, 0.7311111659353473], [0.7845583675276703, 0.7887232687566825, 0.7808607173344968, 0.7011807814573678], [0.7721306773643968, 0.8039713783597632, 0.7523201824527153, 0.7232830861652301]
    ]

# Results from all 33 lasso features without derived columns
all_feat = [
    [0.4898690786712633, 0.7918913539972113, 0.4348693176756019, 0.6251188298715767], [0.7148412858171306, 0.7413451467920957, 0.6968296397058416, 0.662691012993338], [0.6486733685838537, 0.7569520755255891, 0.6054589186998821, 0.6463153971273936], [0.4438749895645553, 0.7278682369676113, 0.40354884480652087, 0.5740911651604136], [0.6964004138350969, 0.7673919001133996, 0.6575529029818845, 0.6338078001223022]
             ]

# Results from the grid search without derived columns
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

# Results from the initial logistic regression using all 50+4 features
initial_feat = [
    [0.6443993149249017, 0.8078082684942527, 0.5886079421332077, 0.7117815140510514], [0.7141702626497349, 0.7844903944880464, 0.684824458559949, 0.7442511108127259], [0.666058311063887, 0.7873562558443744, 0.6247148293685167, 0.7317664020864817], [0.6082090520897839, 0.7697697767277337, 0.5598282391578777, 0.6855099973919342], [0.7943994133679596, 0.805052748318414, 0.785989724849017, 0.7208277362195241]
    ]

# Results from the initial logistic regression using 50 features without derived
initial_feat_noderived = [
    [0.4762414367224223, 0.7923266364473441, 0.423327550648422, 0.6251134618166008], 
    [0.6937582245310022, 0.7459530333766964, 0.6656948054573257, 0.6605611387723452], 
    [0.6924207495664935, 0.7513215660234414, 0.6604458885241513, 0.6460496513974494], 
    [0.4485597518622181, 0.7280965475512767, 0.4072888570679058, 0.5746125263801196], 
    [0.7089146935542591, 0.76646674425197, 0.6748245973812828, 0.6341752480779391]]

feat_list = [final_feat, all_feat,feat1, feat2, feat3, feat4, feat5, initial_feat, initial_feat_noderived]

calcScores(feat_list)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Evaluation

# COMMAND ----------

# Function to save metrics as separate lists

def save_metrics(model_metrics):
    f1_scores = []
    precision_scores = []
    recall_scores = []
    auc_scores = []
    for item in model_metrics:
        f1_scores.append(item[0])
        precision_scores.append(item[1])
        recall_scores.append(item[2])
        auc_scores.append(item[3])
    return f1_scores, precision_scores, recall_scores, auc_scores

# COMMAND ----------

# Get LASSO Validation Metrics
# NOTE this might be  outdated

# Conclusion: The AUC is consistent across all folds for the features they selected except fold 4

# NOTE will need to update validationMetrics to validationMetrics_lasso if lasso is rerun
f1_lasso, precision_lasso, recall_lasso, auc_lasso = save_metrics(validationMetrics)

# Labels for the x-axis
labels = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]

# Create a line plot
plt.plot(labels, auc_lasso, marker='o', linestyle='-', color='b', label="AUC")
plt.plot(labels, precision_lasso, marker='o', linestyle='-', color='r', label="Precision")
plt.plot(labels, recall_lasso, marker='o', linestyle='-', color='g', label="Recall")
plt.plot(labels, f1_lasso, marker='o', linestyle='-', color='m', label="F1")

# Set plot title and axis labels
plt.title('LASSO Validation Evaluation')
plt.xlabel('Folds')
plt.ylabel('Values')

# Show legend
plt.legend()

# Show the plot
plt.show()


# COMMAND ----------

# Items in feature list are defined above. Each feat contains the performance metrics for a feature set across 5 folds

feat_list = [feat1, feat2, feat3, feat4, feat5]

# Initialize dictionaries
average_f1_scores = {}
average_w_precision_scores = {}
average_w_recall_scores = {}
average_AuC_scores = {}

# This averages the evaluation metrics across folds for each feature set
for i in range(len(feat_list)):
    feat_name = f"feat{i+1}"
    validation_metrics = feat_list[i]

    # Transpose the matrix to get lists of metrics
    f1_score, weighted_precision, weighted_recall, AuC = zip(*validation_metrics)

    # Calculate averages
    average_f1_score = sum(f1_score) / len(f1_score)
    average_w_precision = sum(weighted_precision) / len(weighted_precision)
    average_w_recall = sum(weighted_recall) / len(weighted_recall)
    average_AuC = sum(AuC) / len(AuC)

    # Save average scores 
    average_f1_scores[feat_name] = average_f1_score
    average_w_precision_scores[feat_name] = average_w_precision
    average_w_recall_scores[feat_name] = average_w_recall
    average_AuC_scores[feat_name] = average_AuC

# COMMAND ----------

# NOTE this might be outdated

import matplotlib.pyplot as plt
import numpy as np

# List of feature names
features = ["feat1", "feat2", "feat3", "feat4", "feat5"]

# Values for each metric
f1_values = [average_f1_scores[feat] for feat in features]
precision_values = [average_w_precision_scores[feat] for feat in features]
recall_values = [average_w_recall_scores[feat] for feat in features]
auc_values = [average_AuC_scores[feat] for feat in features]

# Bar width
bar_width = 0.2

# Set up positions for each bar
index = np.arange(len(features))
index_f1 = index
index_precision = index + bar_width
index_recall = index + 2 * bar_width
index_auc = index + 3 * bar_width

# Create bar chart
plt.bar(index_f1, f1_values, color='blue', width=bar_width, label='F1')
plt.bar(index_precision, precision_values, color='green', width=bar_width, label='Precision')
plt.bar(index_recall, recall_values, color='orange', width=bar_width, label='Recall')
plt.bar(index_auc, auc_values, color='red', width=bar_width, label='AUC')

# Set labels and title
plt.xlabel('Features')
plt.ylabel('Values')
plt.title('Feature Set Metrics')
plt.xticks(index + 1.5 * bar_width, features)
plt.legend()

# Show the plot
plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

# Conclusions: The features selected from fold 3 performed the best across all folds when used in logistic regression

# List of feature names
features = ["feat1", "feat2", "feat3", "feat4", "feat5"]

# Values for each metric - does not include derived features
f1_values = [average_f1_scores[feat] for feat in features]
auc_values = [average_AuC_scores[feat] for feat in features]

# Add scores for 33 feature lasso without derived features
f1_values.append(0.59873182729438) 
auc_values.append(0.6284048410550048)

# Bar width
bar_width = 0.4

# Set up positions for each bar
index = np.arange(len(features)+1)
index_f1 = index
index_auc = index + bar_width

# Create bar chart
plt.bar(index_f1, f1_values, color='blue', width=bar_width, label='F1')
plt.bar(index_auc, auc_values, color='orange', width=bar_width, label='AUC')

# Set labels and title
plt.xlabel('Feature Sets')
plt.ylabel('Values')
plt.suptitle('Feature Set AUC and F1')
plt.title('Metrics Averaged Across Folds')
plt.xticks(index + 0.5 * bar_width, ["Set 1", "Set 2", "Set 3", "Set 4", "Set 5", "All"])
plt.legend()

# Show the plot
plt.show()


# COMMAND ----------

# Get Features from LF3

# LF3 features
LF3_features = [0,1,2,6,8,12,15,22,23,25,26,32,37,39,40,41,43,46,50,51]

feat_idx_map = pd.DataFrame(folds[0][0].schema["features"].metadata["ml_attr"]["attrs"]["binary"]+folds[0][0].schema["features"].metadata["ml_attr"]["attrs"]["numeric"]).sort_values("idx")

feat_idx_map[feat_idx_map['idx'].isin(LF3_features)]

# COMMAND ----------

# Compare validation metrics of original log w derived VS LF3 w derived VS original wo derived

import numpy as np
import matplotlib.pyplot as plt

# Data
LF3_f1 = 0.742
LF3_auc = 0.723
init_f1 = 0.686
init_auc = 0.719
init_noderived_f1 = 0.604
init_noderived_auc = 0.628

# Bar positions
bar_positions = np.arange(3)  

# Bar heights
bar_heights_f1 = [init_noderived_f1, init_f1, LF3_f1]
bar_heights_auc = [init_noderived_auc, init_auc, LF3_auc]

# Bar width
bar_width = 0.35

# Create a grouped bar chart
bars1 = plt.bar(bar_positions - bar_width/2, bar_heights_f1, bar_width, label='F1', color='blue')
bars2 = plt.bar(bar_positions + bar_width/2, bar_heights_auc, bar_width, label='AUC', color='orange')

# Set labels and title
plt.xlabel('Models')
plt.ylabel('Values')
plt.suptitle('Scores for Initial and Best LASSO Models')
plt.title('Avg across Validation Folds')

# Set x-axis ticks and labels
plt.xticks(bar_positions, ['Init WO Derived', 'Init W Derived', 'Best Lasso'])

# Add a legend
plt.legend()

# Add numerical values on top of the bars
for bar1, bar2 in zip(bars1, bars2):
    plt.text(bar1.get_x() + bar1.get_width() / 2 - 0.08, bar1.get_height() + 0.01, f'{bar1.get_height():.3f}', ha='center', va='bottom', color='black', fontweight='bold')
    plt.text(bar2.get_x() + bar2.get_width() / 2 + 0.08, bar2.get_height() + 0.01, f'{bar2.get_height():.3f}', ha='center', va='bottom', color='black', fontweight='bold')

# Show the plot
plt.show()

# COMMAND ----------

# Calculate averages of training metrics to see if overfitting occurred

# Results from the initial logistic regression using 50 features without derived
initial_feat_noderived_train = [
    [0.5883271765959202, 0.7610788333399522, 0.5456233648447233, 0.6676977053914612], [0.6970358091009783, 0.78555090181029, 0.6529211613269746, 0.6565222141548691], [0.6756213311189279, 0.7782191227570713, 0.6292907575848893, 0.6526719162760372], [0.6658307121683524, 0.7854808947486376, 0.6192475045757307, 0.6818176145720058], [0.6142186443203934, 0.7700020935009995, 0.5692138290549242, 0.6773173510794839]
    ]

# Results from the initial logistic regression using all 50+4 features
initial_feat_train = [
    [0.6877065521081677, 0.7851085529903671, 0.6533194527023393, 0.7448139056384733], [0.7809191915032684, 0.8161512528607194, 0.7596381967706376, 0.7369388275082458], [0.7684241751708129, 0.8106260693829788, 0.7447350681645991, 0.7419997663837345], [0.7504802260124205, 0.8105335171327716, 0.7214980093212824, 0.7544714217083646], [0.7096522574607212, 0.7945043999278255, 0.6765172085765131, 0.752800379677295]
    ]

# Results from the 20+4 regression (20 = feat3 = best performing lasso feature set)
final_feat_train = [
    [0.6795806571951428, 0.7813950775223373, 0.6440946166875539, 0.7384601858722438], [0.7820126564369381, 0.8152990201165846, 0.761529864486273, 0.7343319405288821], [0.768729672450981, 0.8100547618510459, 0.7453345753116433, 0.7399118375186631], [0.7492645827601563, 0.8085557120925304, 0.720300394361945, 0.7496799644596933], [0.705849033036964, 0.7920780900034714, 0.6721509271810558, 0.7490757547850648]
    ]


feat_list_train = [initial_feat_noderived_train, initial_feat_train, final_feat_train]

calcScores(feat_list_train)

# COMMAND ----------

# Compare training metrics of original log w derived VS LF3 w derived VS original wo derived

import numpy as np
import matplotlib.pyplot as plt

# Data
init_noderived_f1 = 0.648
init_noderived_auc = 0.667
init_f1 = 0.739
init_auc = 0.746
LF3_f1 = 0.737
LF3_auc = 0.742

# Bar positions
bar_positions = np.arange(3)  

# Bar heights
bar_heights_f1 = [init_noderived_f1, init_f1, LF3_f1]
bar_heights_auc = [init_noderived_auc, init_auc, LF3_auc]

# Bar width
bar_width = 0.35

# Create a grouped bar chart
bars1 = plt.bar(bar_positions - bar_width/2, bar_heights_f1, bar_width, label='F1', color='blue')
bars2 = plt.bar(bar_positions + bar_width/2, bar_heights_auc, bar_width, label='AUC', color='orange')

# Set labels and title
plt.xlabel('Models')
plt.ylabel('Values')
plt.suptitle('Scores for Initial and Best LASSO Models')
plt.title('Avg across Training Folds')

# Set x-axis ticks and labels
plt.xticks(bar_positions, ['Init WO Derived', 'Init W Derived', 'Best Lasso'])

# Add a legend
plt.legend(loc='center right')

# Add numerical values on top of the bars
for bar1, bar2 in zip(bars1, bars2):
    plt.text(bar1.get_x() + bar1.get_width() / 2 - 0.08, bar1.get_height() + 0.01, f'{bar1.get_height():.3f}', ha='center', va='bottom', color='black', fontweight='bold')
    plt.text(bar2.get_x() + bar2.get_width() / 2 + 0.08, bar2.get_height() + 0.01, f'{bar2.get_height():.3f}', ha='center', va='bottom', color='black', fontweight='bold')

# Show the plot
plt.show()


# COMMAND ----------

