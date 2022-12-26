## Machine Learning on Snowflake

### snowflake is one of the leading data platforms out there. In this article, we will explore the capabilities of its' snowpark python library.

Throughout this article, you will learn how to use,
- snowpark-python functionalities for primary data preprocessing
- Train and deploy machine learning models in Snowflake

Define UDFs in a pythonic way and deploy them in snowflake
Topics covered in this article | Image by authorIf you are willing to follow along with the tutorial, you should have an Anaconda integration-enabled snowflake account. Otherwise, you must sign up for a free snowflake trial account and configure it as described here.
At first glance, snowpark is a machine learning and data science framework that offers the power of SQL within Python flexibility. Sometimes, this is similar to the Apache spark framework. However, this provides a pervasive framework for our machine learning and data science projects. Before trying anything in this article, you should establish a connection between python and Snowflake. You can refer to my code repo for code samples. Let's create a database connection.

``` python 
from snowflake.snowpark.session import Session

accountname = "********" # your accountname
username = "**********" #your snowflake username
password = "*************" #snowflake password

connection_parameters = {
    "account": accountname,
    "user": username,
    "password": password,
    "role": "ACCOUNTADMIN"
}

def snowflake_connector():
    try:
        session = Session.builder.configs(connection_parameters).create()
        print("connection successful!")
    except:
        raise ValueError("error while connecting with db")
    return session

#define a session
session = snowflake_connector()
```

Now we can start the primary data preprocessing part. Instead of preprocessing with Pandas DataFrame, I will do this with the snowpark side. Here I'm going to use the COVID-19 Dataset, which is available in the Kaggle under CC0: Public Domain. I have already loaded this Dataset as a Snowflake table. Since it's not the primary goal of this article, I'm skipping that part. You can load the Dataset as described in this article's GitHub repo. Let's read the table.

``` python 
snowpark_df = session.table("COVID19_RECORDS")

print(type(snowpark_df) # snowflake.snowpark.table.Table
print(f"Size of the table object: {(sys.getsizeof(snowpark_df)/1e6)} MB")
#'Size of the table object: 4.8e-05 MB'
Above snowpark_df is a lazily-evaluated table; hence It won't consume much memory like pandas data frames. But we can apply any transformations aggregations and much more as we did with pandas.
snowpark_df.schema.fields

# [StructField('USMER', LongType(), nullable=True),
#  StructField('MEDICAL_UNIT', LongType(), nullable=True),
#  StructField('SEX', LongType(), nullable=True),
#  StructField('PATIENT_TYPE', LongType(), nullable=True),
#  StructField('DATE_DIED', StringType(), nullable=True),
#  StructField('INTUBED', LongType(), nullable=True),
#  StructField('PNEUMONIA', LongType(), nullable=True),
#  StructField('AGE', LongType(), nullable=True),
#  StructField('PREGNANT', LongType(), nullable=True),
#  StructField('DIABETES', LongType(), nullable=True),
#  StructField('COPD', LongType(), nullable=True),
#  StructField('ASTHMA', LongType(), nullable=True),
#  StructField('INMSUPR', LongType(), nullable=True),
#  StructField('HIPERTENSION', LongType(), nullable=True),
#  StructField('OTHER_DISEASE', LongType(), nullable=True),
#  StructField('CARDIOVASCULAR', LongType(), nullable=True),
#  StructField('OBESITY', LongType(), nullable=True),
#  StructField('RENAL_CHRONIC', LongType(), nullable=True),
#  StructField('TOBACCO', LongType(), nullable=True),
#  StructField('CLASIFFICATION_FINAL', LongType(), nullable=True),
#  StructField('ICU', LongType(), nullable=True)]
```

There are 1,048,575 unique records and 21 columns in the Dataset. Let's do some fundamental analysis. First, let's define the target variable as follow. As per the description of the Dataset, the 1,2 and 3 values in CLASSIFICATION_FINAL the column represent the positive cases, and the rest represent the negative cases. Let's define a new column called TARGET by applying the above logic. The equivalent SQL logic will be,
``` sql 
SELECT
    "USMER",
    "MEDICAL_UNIT",
    "SEX",
    "PATIENT_TYPE",
    "DATE_DIED",
    "INTUBED",
    "PNEUMONIA",
    "AGE",
    "PREGNANT",
    "DIABETES",
    "COPD",
    "ASTHMA",
    "INMSUPR",
    "HIPERTENSION",
    "OTHER_DISEASE",
    "CARDIOVASCULAR",
    "OBESITY",
    "RENAL_CHRONIC",
    "TOBACCO",
    "CLASIFFICATION_FINAL",
    "ICU",
    CASE
        WHEN ("CLASIFFICATION_FINAL" < 4 :: INT) THEN 1 :: INT
        ELSE 0 :: INT
    END AS "TARGET"
FROM
    COVID19_RECORDS
```

Since we are working with snowpark API, let's create this with snowpark.

``` python 
import snowflake.snowpark.functions as F

snowpark_df.with_column('TARGET', F.when(F.col('CLASIFFICATION_FINAL')
                        < 4, 1).otherwise(0))
Let's see our target distribution.
snowpark_df\
.group_by("TARGET").count().to_pandas().set_index("TARGET")\
.plot.bar()

plt.title("Target distribution",fontweight='semibold')
plt.show()
Let's create one more plot.
snowpark_df\
.select('AGE').to_pandas()\
.plot.hist(bins=100,alpha=0.5)

plt.title("Age distribution",fontweight='semibold')
plt.show()
Let's find the relationship between the Age variable and the target variable.
snowpark_df = snowpark_df.with_column(
    "AGE_BKT",
    F.when(F.col("AGE") < 21, "YOUNG").otherwise(
        F.when(F.col("AGE") < 49, "ADULT").otherwise("OLD ADULT")
    ),
)

age_bkt_df = snowpark_df.select(
    F.col("AGE_BKT"),
    F.when((F.col("AGE_BKT")=='YOUNG') & (F.col("TARGET")==1),1).otherwise(0).as_("YOUNG_"),
    F.when((F.col("AGE_BKT")=='ADULT') & (F.col("TARGET")==1),1).otherwise(0).as_("ADULT_"),
    F.when((F.col("AGE_BKT")=='OLD ADULT') & (F.col("TARGET")==1),1).otherwise(0).as_("OLD_ADULT_")
)

age_bkt_df.group_by(F.col("AGE_BKT")).count().show()

# -----------------------
# |"AGE_BKT"  |"COUNT"  |
# -----------------------
# |OLD ADULT  |342413   |
# |ADULT      |628554   |
# |YOUNG      |77608    |
# -----------------------

age_bkt_df.select(
    ((F.sum("YOUNG_") * 100 ) / F.count("YOUNG_")).as_("YOUNG % OF CASES"),
    ((F.sum("ADULT_") * 100) / F.count("ADULT_")).as_("ADULT % OF CASES"),
    ((F.sum("OLD_ADULT_") * 100) / F.count("OLD_ADULT_")).as_("OLD_ADULT % OF CASES")
).show()

# --------------------------------------------------------------------
# |"YOUNG % OF CASES"  |"ADULT % OF CASES"  |"OLD_ADULT % OF CASES"  |
# --------------------------------------------------------------------
# |1.534463            |20.877858           |14.969745               |
# --------------------------------------------------------------------
```

After completing our analysis, we can save the transformed Dataset as a new Snowflake table using the following way.

``` python 
snowpark_df.write.save_as_table(
    table_name='COVID19_RECORDS_PROCESSED',
    mode='overwrite'
)
```

Alright, now we have preprocessed Dataset. Let's start the model training phase.

``` python 
# read the table 
train_data = session.table("COVID19_RECORDS_PROCESSED")

#create the stage for storing the ML models
session.sql('CREATE OR REPLACE STAGE ML_MODELS').show()
```

We can use two different approaches to train and deploy models in Snowflake.
We can train the model locally, upload it to a stage and load it from the stage when the UDF is called.
We can define SPROC, which can train the model and save the trained model into the Snowflake stage when the SPROC is called. Here we'll need a separate UDF for the inferencing part.

In this article, we will explore both methods above.
Train the model locally, upload it to a stage and load it from the stage
First, we have to define the function for training the model locally.

``` python 
def train_model_locally(train:snowflake.snowpark.table.Table):
    from sklearn.tree import DecisionTreeClassifier
    
    #convert into pd dataframes
    
    train = train.to_pandas()
    
    xtrain,ytrain = train.drop('TARGET',axis=1),train['TARGET']
    
    model = DecisionTreeClassifier()
    model.fit(xtrain,ytrain)
    
    return model

#let's train the DT model
model = train_model_locally(train_data_sf)

#save the model
import joblib
joblib.dump(model, 'predict_risk_score.joblib')

#upload into the ML_MODELS SNowfla
session.file.put(
    "predict_risk_score.joblib", "@ML_MODELS", auto_compress=False, overwrite=True
)
```

Similar to other machine learning pipelines, we need to define library dependencies.

``` python 
session.clear_imports()
session.clear_packages()

#Register above uploded model as import of UDF
session.add_import("@ML_MODELS/predict_risk_score.joblib")

#map packege dependancies
session.add_packages("joblib==1.1.0", "scikit-learn==1.1.1", "pandas==1.3.2")
```

Let's define the UDF. Inside the UDF, it should load the model from the stage and then use it for the inferencing.

``` python 
from snowflake.snowpark.types import PandasSeries, PandasDataFrame

def read_file(filename):
    import joblib
    import sys
    import os
    
    #where all imports located at
    import_dir = sys._xoptions.get("snowflake_import_directory")

    if import_dir:
        with open(os.path.join(import_dir, filename), 'rb') as file:
            m = joblib.load(file)
            return m

#register UDF
@F.udf(name = 'predict_risk_score', is_permanent = True, replace = True, stage_location = '@ML_MODELS')
def predict_risk_score(ds: PandasSeries[dict]) -> PandasSeries[float]:
    
    # later we will input train data as JSON object
    # hance, we have to convert JSON object as pandas DF
    df = pd.io.json.json_normalize(ds)[feature_cols]
    pipeline = read_file('predict_risk_score.joblib')
    return pipeline.predict_proba(df)[:,1]
```
Now we have successfully registered our UDF in Snowflake. You can verify it using the following way.

``` python 
session.sql("DESC FUNCTION PREDICT_RISK_SCORE()").show()

# ------------------------------------------------------------------------
# |"property"       |"value"                                             |
# ------------------------------------------------------------------------
# |signature        |()                                                  |
# |returns          |FLOAT                                               |
# |language         |PYTHON                                              |
# |null handling    |CALLED ON NULL INPUT                                |
# |volatility       |VOLATILE                                            |
# |body             |                                                    |
# |                 |import pickle                                       |
# |                 |                                                    |
# |                 |func = pickle.loads(bytes.fromhex('800595050400...  |
# |                 |# The following comment contains the UDF source...  |
# |                 |# import pandas as pd                               |
# |                 |# def read_file(filename):                          |
# |                 |#     import joblib                                 |
# |                 |#     import sys                                    |
# |                 |#     import os                                     |
# |                 |#                                                   |
# |                 |#     import_dir = sys._xoptions.get("snowflake...  |
# |                 |#     if import_dir:                                |
# |                 |#         with open(os.path.join(import_dir, fi...  |
# |                 |#             m = joblib.load(file)                 |
# |                 |#             return m                              |
# |                 |# @F.udf(name = 'predict_risk_score', is_perman...  |
# |                 |# def predict_risk_score(*args) -> PandasSeries...  |
# |                 |#     df = pd.DataFrame([args])                     |
# |                 |#     pipeline = read_file('predict_risk_score....  |
# |                 |#     return pipeline.predict_proba(df)[:,1]        |
# |                 |#                                                   |
# |                 |# func = predict_risk_score                         |
# |                 |#                                                   |
# |                 | *********RESULTS TRUNCATED**************           |
# ------------------------------------------------------------------------
```
Let's use UDF for inferencing.

``` python 
# `test_data_sf` is a fraction of `train_data`
test_data_sf.with_column(
    'PREDICTION', 
    predict_risk_score(F.object_construct('*')))\
.select("TARGET","PREDICTION").show(20)

# ---------------------------------
# |"TARGET"  |"PREDICTION"        |
# ---------------------------------
# |1         |0.8333333333333334  |
# |1         |0.0                 |
# |1         |1.0                 |
# |1         |1.0                 |
# |1         |0.3333333333333333  |
# |0         |0.0                 |
# |1         |0.4                 |
# |0         |0.5                 |
# |1         |0.421875            |
# ---------------------------------

#similary, you can use below SQL as well.
```
``` sql 
select
    target,
    predict_risk_score(object_construct(*)) as predictions
from
    COVID19_RECORDS_PROCESSED
limit
    100;
```
Define train and inferencing procs/UDFs
This method will create a stored procedure for training the model and UDF for inferencing the model. You may refer to the diagram below for more insights.
Let's define the stored procedure. At first, we will implement the Python function, and we can convert it to the Snowflake stored procedure in later steps.

``` python 
def train_dt_procedure(
    session: Session,
    training_table: str,
    feature_cols: list,
    target_col: str,
    model_name: str,
) -> T.Variant:
    
    """
    This will be our training procedure. Later we will register this as snowflake procedure.
    
    training_table: snowflake table name to be used for training task
    feature_cols: list of columns to be used in training
    target_col: target column to be used
    model_name: model name to used for model saving purpose
    
    """

    #convert as pandas DF, rest of the steps similar to the local model training and saving.
    local_training_data = session.table(training_table).to_pandas()

    from sklearn.tree import DecisionTreeClassifier

    X = local_training_data[feature_cols]
    y = local_training_data[target_col]

    model = DecisionTreeClassifier()
    model.fit(X, y)
    
    #do what ever you want to do with model, even the hyperparameter tuning..
    # here I'll get feature importance
    feat_importance = pd.DataFrame(
        model.feature_importances_, feature_cols, columns=["FeatImportance"]
    ).to_dict()

    from joblib import dump

    dump(model, "/tmp/" + model_name)
    session.file.put(
        "/tmp/" + model_name, "@ML_MODELS", auto_compress=False, overwrite=True
    )
    return feat_importance
Let's register the above Python function as a stored procedure.
sproc_train_dt_model = session.sproc.register(
                    func=train_dt_procedure, 
                    name='sproc_train_dt_model', 
                    is_permanent=True, 
                    replace=True, 
                    stage_location='@ML_MODELS', 
                    packages=[
                        'snowflake-snowpark-python',
                        'scikit-learn',
                        'joblib']
)
```

Now we can use the procedure `SPROC_TRAIN_DT_MODEL()` as follows.

``` python 
train_data = session.table("COVID19_RECORDS_PROCESSED")

#create train and test dataframes
train_data_pd,test_data_pd = train_test_split(
                                        train_data_pd,
                                        stratify=train_data_pd['TARGET'],
                                        test_size=0.1
)

# writing as tempoary tables for mode training and inferencing part
session.write_pandas(
    train_data_pd,
    table_name="TRAIN_DATA_TMP",
    auto_create_table=True,
    table_type="temporary",
)
session.write_pandas(
    test_data_pd,
    table_name="TEST_DATA_TMP",
    auto_create_table=True,
    table_type="temporary",
)

train_data_pd = train_data.to_pandas()

feature_cols = train_data.columns
feature_cols.remove('TARGET')
target_col = 'TARGET'
model_name = 'decisiontree.model' # How model should be saved in stage

model_response = sproc_train_dt_model('TRAIN_DATA_TMP', 
                                            feature_cols, 
                                            target_col,
                                            model_name, 
                                            session=session
                                           )

print(model_response)

# {
#   "FeatImportance": {
#     "AGE": 0.4543249401305732,
#     "ASTHMA": 0.029003830541684678,
#     "CARDIOVASCULAR": 0.025649097586968667,
#     "COPD": 0.019300936592021863,
#     "DIABETES": 0.059273293874405074,
#     "HIPERTENSION": 0.05885196748765571,
#     "INMSUPR": 0.0232534703448427,
#     "INTUBED": 0.026365011429648998,
#     "MEDICAL_UNIT": 0.08804779552309593,
#     "OBESITY": 0.02991724846285235,
#     "OTHER_DISEASE": 0.026840169399286344,
#     "PATIENT_TYPE": 0,
#     "PNEUMONIA": 0.04225497414608237,
#     "PREGNANT": 0.012929499812685114,
#     "RENAL_CHRONIC": 0.015894267526361774,
#     "SEX": 0,
#     "TOBACCO": 0.028563364646896985,
#     "USMER": 0.059530132494938236
#   }
# }

#plot feature importance
feature_coefficients = pd.DataFrame(eval(model_response))

feature_coefficients\
.sort_values(by='FeatImportance',ascending=False)\
.plot\
.bar(y='FeatImportance', figsize=(12,5))
plt.show()
```

We can define the UDF as follows. This function is similar to the previous one.

``` python 
def udf_predict_risk_score(*args) -> float:
    import os
    import sys
    from joblib import load
    
    IMPORT_DIRECTORY_NAME = "snowflake_import_directory"
    import_dir = sys._xoptions[IMPORT_DIRECTORY_NAME]
    model_name = 'decisiontree.model'
    model = load(import_dir+model_name)
    
    #unlike previous JSON object, this will be a array, hence no need to
    # decode the input
    scored_data = model.predict(pd.DataFrame([args]))[0]

    return scored_data
```
Finally, registering the UDF.

``` python 
udf_risk_score_model = session.udf.register(
                            func=udf_predict_risk_score, 
                            name="udf_risk_score_model", 
                            stage_location='@ML_MODELS',
                            input_types=[T.FloatType()]*len(feature_cols),
                            return_type = T.FloatType(),
                            replace=True, 
                            is_permanent=True, 
                            imports=['@ML_MODELS/decisiontree.model'],
                            packages=['scikit-learn==1.1.1','pandas','joblib'], 
                            session=session
)
```
Alright, it's time to get predictions for our validation dataset. Here I am doing it with Snowflake editor.

``` sql 
SELECT
    "TARGET",
    udf_risk_score_model(
        "USMER",
        "MEDICAL_UNIT",
        "SEX",
        "PATIENT_TYPE",
        "INTUBED",
        "PNEUMONIA",
        "AGE",
        "PREGNANT",
        "DIABETES",
        "COPD",
        "ASTHMA",
        "INMSUPR",
        "HIPERTENSION",
        "OTHER_DISEASE",
        "CARDIOVASCULAR",
        "OBESITY",
        "RENAL_CHRONIC",
        "TOBACCO"
    ) AS "PREDICTION"
FROM
    COVID19_RECORDS_PROCESSED limit 100;
```
## Conclusion
While snowpark offers a comprehensive platform for our machine learning tasks, it has a few issues at the time of writing this article. As an example, PyTorch still needs to be supported by a snowpark. Also, only selected packages are available in conda; if we want to use other packages, such as catboost, we must import them manually into our environment as described here.
Thanks for reading! Connect with me on LinkedIn.