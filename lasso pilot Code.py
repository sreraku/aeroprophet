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
# MAGIC Read in Provided Data

# COMMAND ----------

try:
    df_OTPW_60M = spark.read.parquet(f"{team_blob_url}/OTPW_60M")
except:
    try:
        df_OTPW_60M = spark.read.parquet(f"{team_blob_url}/OTPW_60M")
    
    except:
        df_OTPW_60M = spark.read.option("header", True) \
                                .option("compression", "gzip") \
                                .option("delimiter",",") \
                                .csv(f"{mids261_mount_path}/OTPW_60M")
                        
        df_OTPW_60M.write.parquet(f"{team_blob_url}/OTPW_60M")

# COMMAND ----------

# MAGIC %md
# MAGIC Read in Our Joined Data

# COMMAND ----------

# loading in smaller dataset 
df_60M = spark.read.parquet(f"{team_blob_url}/df_FS_3M").cache()

# COMMAND ----------

# MAGIC %md
# MAGIC Drop Columns From Our Join That Don't Appear in Provided Data

# COMMAND ----------

df_60M = df_60M.drop(*[c for c in df_60M.columns if c not in [*df_OTPW_60M.columns,'depart_UTC','Depart_3hrs_before']])

df_60M  = df_60M \
    .withColumnRenamed("year_flights", "YEAR") \
    .withColumnRenamed("depart_UTC", "sched_depart_date_time_UTC") \
    .withColumnRenamed("Depart_3hrs_before", "two_hours_prior_depart_UTC") # adjust this later when we can replace all other references in the code

# COMMAND ----------

# MAGIC %md
# MAGIC Check for duplicates

# COMMAND ----------

# MAGIC %md
# MAGIC Drop Duplicates

# COMMAND ----------

df_60M = df_60M.dropDuplicates()

# COMMAND ----------

# MAGIC %md
# MAGIC Review fields, count Null + NaN Values, and Determine Datatypes

# COMMAND ----------

pd.set_option('display.max_rows', None)

# COMMAND ----------

# add feature that checks # of flights in last 24 hours of tail number that were delayed
# add feature flights delayed at airpot in last 24 hours
from pyspark.sql.window import Window
df_60M = df_60M.withColumn('sch_dep_time_linux', F.unix_timestamp("sched_depart_date_time_UTC"))

#creating # of plane delays in last 24 hours
windowSpec = Window().partitionBy('TAIL_NUM').orderBy('sch_dep_time_linux')\
    .rangeBetween(-86400,-10800)
df_60M = df_60M.withColumn('Plane_Delays_last_24H', F.sum(F.col("DEP_DEL15").cast("int")).over(windowSpec))

#creating airport delay last 24 hr feature

#finding number of flights delayed at airport in last 24 hours
windowSpec = Window().partitionBy('ORIGIN').orderBy('sch_dep_time_linux')\
    .rangeBetween(-86400,-10800)
df_60M = df_60M.withColumn('airport_delays_in_previous_24_hours', F.sum(F.col("DEP_DEL15").cast("int")).over(windowSpec))

df_60M = df_60M.fillna(0,subset=['Plane_Delays_last_24H','airport_delays_in_previous_24_hours'])


# COMMAND ----------

columns_to_drop = [
    'MonthlyStationPressure', #this first set of variables has very little data
    'MonthlyDepartureFromNormalMinimumTemperature',
    'MonthlyMeanTemperature',
    'MonthlyMinSeaLevelPressureValue',
    'MonthlyMinSeaLevelPressureValueDate',
    'MonthlyMinSeaLevelPressureValueTime',
    'MonthlyMinimumTemperature',
    'MonthlySeaLevelPressure',
    'MonthlyDaysWithGT32Temp',
    'MonthlyDaysWithGT010Precip',
    'MonthlyTotalLiquidPrecipitation',
    'MonthlyTotalSnowfall',
    'MonthlyWetBulb',
    'AWND',
    'CDSD',
    'CLDD',
    'DSNW',
    'MonthlyDaysWithGT90Temp',
    'MonthlyDaysWithLT0Temp',
    'MonthlyDaysWithLT32Temp',
    'MonthlyGreatestSnowfallDate',
    'MonthlyDewpointTemperature',
    'MonthlyGreatestPrecip',
    'MonthlyGreatestPrecipDate',
    'MonthlyGreatestSnowDepth',
    'MonthlyGreatestSnowDepthDate',
    'MonthlyGreatestSnowfall',
    'MonthlyMaxSeaLevelPressureValue',
    'MonthlyDepartureFromNormalAverageTemperature',
    'MonthlyMaxSeaLevelPressureValueDate',
    'MonthlyMaxSeaLevelPressureValueTime',
    'MonthlyMaximumTemperature',
    'MonthlyDepartureFromNormalMaximumTemperature',
    'MonthlyDepartureFromNormalHeatingDegreeDays',
    'MonthlyDepartureFromNormalCoolingDegreeDays',
    'HDSD',
    'HTDD',
    'NormalsCoolingDegreeDay',
    'ShortDurationPrecipitationValue010',
    'MonthlyAverageRH',
    'MonthlyDaysWithGT001Precip',
    'ShortDurationPrecipitationValue180',
    'ShortDurationPrecipitationValue150',
    'ShortDurationPrecipitationValue120',
    'ShortDurationPrecipitationValue100',
    'ShortDurationPrecipitationValue080',
    'ShortDurationPrecipitationValue060',
    'ShortDurationPrecipitationValue045',
    'ShortDurationPrecipitationValue030',
    'NormalsHeatingDegreeDay',
    'ShortDurationPrecipitationValue020',
    'ShortDurationPrecipitationValue015',
    'MonthlyDepartureFromNormalPrecipitation',
    'ShortDurationPrecipitationValue005',
    'ShortDurationEndDate045',
    'ShortDurationEndDate005',
    'ShortDurationEndDate010',
    'ShortDurationEndDate180',
    'ShortDurationEndDate020',
    'ShortDurationEndDate030',
    'ShortDurationEndDate015',
    'ShortDurationEndDate060',
    'ShortDurationEndDate080',
    'ShortDurationEndDate100',
    'ShortDurationEndDate120',
    'ShortDurationEndDate150',
    'DailyWeather',
    'DailySnowDepth',
    'DailySnowfall',
    'DailyAverageSeaLevelPressure',
    'DailyAverageWetBulbTemperature',
    'DailyAverageDewPointTemperature',
    'DailyAverageRelativeHumidity',
    'DailyDepartureFromNormalAverageTemperature',
    'DailyPeakWindDirection',
    'DailyPeakWindSpeed',
    'DailyAverageStationPressure',
    'DailyHeatingDegreeDays',
    'DailyCoolingDegreeDays',
    'DailyAverageDryBulbTemperature',
    'DailyMaximumDryBulbTemperature',
    'DailyMinimumDryBulbTemperature',
    'DailyAverageWindSpeed',
    'DailySustainedWindDirection',
    'DailyPrecipitation',
    'DailySustainedWindSpeed',
    'Sunrise',
    'Sunset',
    'LONGEST_ADD_GTIME',
    'TOTAL_ADD_GTIME',
    'FIRST_DEP_TIME', 
    'DEP_TIME', # below variables not know until later
    'DEP_DELAY_NEW',
    'DEP_DELAY_GROUP',
    'DEP_TIME_BLK',
    'TAXI_OUT',
    'WHEELS_OFF',
    'WHEELS_ON',
    'TAXI_IN',
    'ARR_TIME',
    'ARR_DELAY',
    'ARR_DELAY_NEW',
    'ARR_DEL15',
    'ARR_DELAY_GROUP',
    'ARR_TIME_BLK',
    'CANCELLED',
    'CANCELLATION_CODE',
    'DIVERTED',
    'AIR_TIME',
    'CARRIER_DELAY',
    'WEATHER_DELAY',
    'NAS_DELAY',
    'SECURITY_DELAY',
    'LATE_AIRCRAFT_DELAY']

# Dropping the columns from the DataFrame
df_60M = df_60M.drop(*columns_to_drop)

# COMMAND ----------

# MAGIC %md
# MAGIC Convert each field to the proper datatypes

# COMMAND ----------

import ast
from dateutil.parser import parse
from pyspark.sql.types import IntegerType, FloatType, StringType, TimestampType

type_dict = {}
for c, dtype in df_60M.dtypes:
    if dtype == 'string':
        value = df_60M.select(c).filter(col(c).isNotNull()).first()[c]

        try:
            literal_value = ast.literal_eval(value)
            if isinstance(literal_value, int):
                type_dict[c] = IntegerType()
            elif isinstance(literal_value, float):
                type_dict[c] = FloatType()
        except:
            try:
                parse(value)
                type_dict[c] = TimestampType()
            except:
                continue

for c in type_dict:
     df_60M = df_60M.withColumn(c, col(c).cast(type_dict[c]))
    

# COMMAND ----------

# MAGIC %md
# MAGIC Add new features

# COMMAND ----------

df_60M.createOrReplaceTempView("df_60M_view")

df_60M = spark.sql("""
    SELECT t1.*,
           FIRST_VALUE(t2.DEP_DELAY) OVER (
               PARTITION BY t1.TAIL_NUM, t1.ORIGIN_AIRPORT_ID, t1.sched_depart_date_time_UTC
               ORDER BY t2.sched_depart_date_time_UTC DESC
           ) AS previous_flight_delay
    FROM df_60M_view t1
    LEFT JOIN (
        SELECT TAIL_NUM, DEST_AIRPORT_ID, sched_depart_date_time_UTC, DEP_DELAY, FL_DATE
        FROM df_60M_view
    ) t2
    ON t1.TAIL_NUM = t2.TAIL_NUM
    AND t1.ORIGIN_AIRPORT_ID = t2.DEST_AIRPORT_ID
    AND t1.two_hours_prior_depart_UTC > t2.sched_depart_date_time_UTC
    AND (DATEDIFF(t1.FL_DATE, t2.FL_DATE) BETWEEN -1 AND 1)
""")


# COMMAND ----------

# Drop any columns that are no longer needed
df_60M = df_60M.drop('DEP_DELAY', 'ACTUAL_ELAPSED_TIME', 'CRS_ELAPSED_TIME')

# Define UDF
def days_to_nearest_holiday(current_date):
    current_date = current_date.date()
    years = [current_date.year - 1, current_date.year, current_date.year + 1]
    us_holidays = holidays.US(years=years)
    dist_to_holiday = min(abs(current_date - d).days for d in us_holidays.keys())

    return dist_to_holiday

# Register the UDF
days_to_nearest_holiday_udf = F.udf(days_to_nearest_holiday, IntegerType())

# Apply the UDF to create a new column
df_60M = df_60M.withColumn('days_to_nearest_holiday', days_to_nearest_holiday_udf(col('FL_DATE')))

# Convert scheduled dep and arr integers to 'hhmm' format
df_60M = df_60M.withColumn('CRS_DEP_TIME_Str', F.lpad(col('CRS_DEP_TIME'), 4, '0'))
df_60M = df_60M.withColumn('CRS_ARR_TIME_Str', F.lpad(col('CRS_ARR_TIME'), 4, '0'))

# Extract the hour part
df_60M = df_60M.withColumn('CRS_DEP_HOUR', F.substring('CRS_DEP_TIME_Str', 1, 2))
df_60M = df_60M.withColumn('CRS_ARR_HOUR', F.substring('CRS_ARR_TIME_Str', 1, 2))

# Convert these hour columns to integers
df_60M = df_60M.withColumn('CRS_DEP_HOUR', df_60M['CRS_DEP_HOUR'].cast('int'))
df_60M = df_60M.withColumn('CRS_ARR_HOUR', df_60M['CRS_ARR_HOUR'].cast('int'))


# COMMAND ----------

# MAGIC %md
# MAGIC Drop unconverted datetime features

# COMMAND ----------

df_60M = df_60M.drop(
                                'CRS_DEP_TIME_Str',
                                'CRS_ARR_TIME_Str',
                                'CRS_DEP_TIME',
                                'CRS_ARR_TIME',
                                'DATE',
                                'sched_depart_date_time_UTC',
                                'sched_arrival_date_time_UTC', 
                                'four_hours_prior_depart_UTC',
                                'two_hours_prior_depart_UTC',
                                'day_before',
                                'day_after',
                                'holiday_day_before',
                                'holiday_day_after',
                                'WindEquipmentChangeDate')


# COMMAND ----------

# adding feature, state region
def state_to_region(state_abrv):
    """converts state abreviations to geographic region """
    region_mapping = {
    'AL': 'Southeast',
    'AK': 'West',
    'AZ': 'West',
    'AR': 'South',
    'CA': 'West',
    'CO': 'West',
    'CT': 'Northeast',
    'DE': 'Northeast',
    'FL': 'Southeast',
    'GA': 'Southeast',
    'HI': 'West',
    'ID': 'West',
    'IL': 'Midwest',
    'IN': 'Midwest',
    'IA': 'Midwest',
    'KS': 'Midwest',
    'KY': 'South',
    'LA': 'South',
    'ME': 'Northeast',
    'MD': 'Northeast',
    'MA': 'Northeast',
    'MI': 'Midwest',
    'MN': 'Midwest',
    'MS': 'South',
    'MO': 'Midwest',
    'MT': 'West',
    'NE': 'Midwest',
    'NV': 'West',
    'NH': 'Northeast',
    'NJ': 'Northeast',
    'NM': 'West',
    'NY': 'Northeast',
    'NC': 'Southeast',
    'ND': 'Midwest',
    'OH': 'Midwest',
    'OK': 'South',
    'OR': 'West',
    'PA': 'Northeast',
    'RI': 'Northeast',
    'SC': 'Southeast',
    'SD': 'Midwest',
    'TN': 'South',
    'TX': 'South',
    'UT': 'West',
    'VT': 'Northeast',
    'VA': 'Southeast',
    'WA': 'West',
    'WV': 'South',
    'WI': 'Midwest',
    'WY': 'West'
}
    return region_mapping.get(state_abrv, 'Unknown')
state_to_region_udf = udf(state_to_region, StringType())
df_60M = df_60M.withColumn('region',state_to_region_udf(col("ORIGIN_STATE_ABR")))
display(df_60M)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's drop the string variables we don't want since it is intensive to convert them into dummies

# COMMAND ----------

df_60M = df_60M.drop('OP_UNIQUE_CARRIER',
                                'OP_CARRIER',
                                'ORIGIN',
                                'TAIL_NUM',
                                'ORIGIN_CITY_NAME',
                                'ORIGIN_STATE_ABR',
                                'ORIGIN_STATE_NM',
                                'DEST',
                                'DEST_CITY_NAME',
                                'DEST_STATE_ABR',
                                'DEST_STATE_NM',
                                'origin_airport_name',
                                'origin_station_name',
                                'origin_station_id',
                                'origin_iata_code',
                                'origin_icao',
                                'origin_type',
                                'origin_region',
                                'dest_airport_name',
                                'dest_station_name',
                                'dest_station_id',
                                'dest_iata_code',
                                'dest_icao',
                                'dest_type',
                                'dest_region',
                                'STATION',
                                'NAME',
                                'REPORT_TYPE',
                                'HourlyPresentWeatherType',
                                'HourlySkyConditions',
                                'REM',
                                'BackupDirection',
                                'BackupDistanceUnit',
                                'BackupElements',
                                'BackupEquipment',
                                'BackupName',
                                'holiday')

# COMMAND ----------

# MAGIC %md
# MAGIC Review Ints - dropping those that are redundant (e.g. ORIGIN_AIRPORT_SEQ_ID is similar to ORIGIN_AIRPORT_ID)

# COMMAND ----------

df_60M = df_60M.drop('ORIGIN_AIRPORT_SEQ_ID',
                                'ORIGIN_CITY_MARKET_ID',
                                'ORIGIN_STATE_FIPS',
                                'ORIGIN_WAC',
                                'DEST_AIRPORT_SEQ_ID',
                                'DEST_CITY_MARKET_ID',
                                'DEST_STATE_FIPS',
                                'DEST_WAC',
                                'SOURCE',
                                'OP_CARRIER_FL_NUM',
                                'BackupElevation')

# COMMAND ----------

# MAGIC %md
# MAGIC Review Floats

# COMMAND ----------

df_60M = df_60M.drop('LATITUDE',
                               'LONGITUDE',
                               'origin_station_lat',
                               'origin_station_lon',
                               'dest_station_lat',
                               'dest_station_lon',
                               'BackupLatitude',
                               'BackupLongitude',
                               'BackupDistance',
                               'FLIGHTS')

# COMMAND ----------

# MAGIC %md
# MAGIC Drop rows where outcome variable is null and null string columns with an empty string

# COMMAND ----------

df_60M = df_60M.na.drop(subset=["DEP_DEL15"])
df_60M = df_60M.na.fill(value='').cache()

# COMMAND ----------

# MAGIC %md
# MAGIC Impute missing numerical values with the mean (this takes quite a while to run)

# COMMAND ----------

df_60M = df_60M.na.fill(value=0) # just filling with 0 for now because it's much faster

# from pyspark.ml.feature import Imputer

# numerical_cols = [c for c, d_type in df_60M.dtypes if d_type in ['int','float', 'double']]

# imputer = Imputer(inputCols= numerical_cols, outputCols=numerical_cols)
# imputer.setStrategy("mean")

# imputer_model = imputer.fit(df_60M)
# df_60M = imputer_model.transform(df_60M)


# COMMAND ----------

# MAGIC %md
# MAGIC Checkpoint the data

# COMMAND ----------

df_60M.write.mode('overwrite').parquet(f"{team_blob_url}/df_FS_3M_chk2")

# Print what's saved in blog storage
display(dbutils.fs.ls(f"{team_blob_url}"))

# COMMAND ----------

# MAGIC %md
# MAGIC Read the parquet file from checkpoint

# COMMAND ----------

df_60M = spark.read.parquet(f"{team_blob_url}/df_FS_3M_chk2").cache()

# COMMAND ----------

#drop the uncorrelated features
df_60M = df_60M.drop('ELEVATION', 'HourlyPressureChange','HourlyPressureTendency', 'HourlyStationPressure')

# COMMAND ----------

# MAGIC %md
# MAGIC Review Distribution

# COMMAND ----------

# MAGIC %md
# MAGIC Create Dummies, Split & Standardize

# COMMAND ----------

df_60M = df_60M.drop("ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "QUARTER")

# COMMAND ----------

display(df_60M)

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

#add region to dummy source cols in main notebook
dummy_source_columns = ["OP_CARRIER_AIRLINE_ID", "DAY_OF_WEEK", 'region']

# Creating a list to store stages of the Pipeline for one-hot encoding and vector assembly
stages = []

# Process each integer column for one-hot encoding
for c in dummy_source_columns:
    # First convert integer IDs to indexed categories
    indexer = StringIndexer(inputCol=c, outputCol=c + "_indexed") # need to remove this as well
    encoder = OneHotEncoder(inputCol=c + "_indexed", outputCol=c + "_dummy")
    stages += [indexer, encoder]

# Create a list of all feature columns, excluding original ID columns and the target variable
feature_columns = [c + "_dummy" for c in dummy_source_columns] + [c for c in df_60M.columns if c not in [*dummy_source_columns,*[c + '_indexed' for c in dummy_source_columns], 'DEP_DEL15', 'FL_DATE']]

# Add the VectorAssembler to the stages
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
stages += [assembler]

# Create the pipeline
pipeline = Pipeline(stages=stages)

# Fit the pipeline to the data and transform
df_60M_transformed = pipeline.fit(df_60M).transform(df_60M)

# Split the data into training and test sets
train_df = df_60M_transformed.filter(df_60M_transformed["YEAR"] != 2019)
test_df = df_60M_transformed.filter(df_60M_transformed["YEAR"] == 2019)

# Create a pipeline for standardization
std_pipeline = Pipeline(stages=[
    StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
])

# Fit the standardization pipeline to the training data
std_pipeline_model = std_pipeline.fit(train_df)

# Transform both training and test data using the standardization pipeline model
train_df_transformed = std_pipeline_model.transform(train_df)
test_df_transformed = std_pipeline_model.transform(test_df)


# COMMAND ----------

# Save the transformed datasets to blob storage
# train_df_transformed.write.mode("overwrite").parquet(f"{team_blob_url}/df_3M_train_transformed")
# test_df_transformed.write.mode("overwrite").parquet(f"{team_blob_url}/df_3M_test_transformed")

train_df_transformed.write.parquet(f"{team_blob_url}/df_3M_train_transformed")
test_df_transformed.write.parquet(f"{team_blob_url}/df_3M_test_transformed")

# COMMAND ----------

# MAGIC %md
# MAGIC Read in Transformed Parquet

# COMMAND ----------

OTPW_train_transformed = spark.read.parquet(f"{team_blob_url}/df_3M_train_transformed")

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

blocks = make_block_splits(date(2015,1,1),date(2018,12,31),timedelta(days = 200),timedelta(days = 92))
folds = make_folds(OTPW_train_transformed, 'FL_DATE', blocks)
save_folds_to_blob(folds, team_blob_url, "k_folds_cross_val")

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

# MAGIC %md
# MAGIC features to add:
# MAGIC - day of week: categorical
# MAGIC - holiday: categorical
# MAGIC - bad weather
# MAGIC - State region
# MAGIC - busier airport
# MAGIC - weekend
# MAGIC - destination airport weather 2 hours before 
# MAGIC - origin airport weather 2 hours before
# MAGIC - destination airport weather window (10-2 hours before)
# MAGIC - number of flights out of destination airport 2 hours before compared to average 

# COMMAND ----------

