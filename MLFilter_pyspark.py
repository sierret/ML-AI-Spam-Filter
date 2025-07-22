from pyspark.ml import Pipeline
import pandas as pd
import numpy as np
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
import matplotlib.pyplot as plt
from pyspark import sql
from pyspark.sql import SparkSession,types
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

tuneHyperPar=False #if to tune the hyperparameters of the classifiers

def viewDataStats(data):
    if (isinstance(data, pd.DataFrame)):
        total_nan = data.isna().sum().sum()
        print("Data Dimensions : %d rows, %d columns" % (data.shape[0], data.shape[1]))
        print("Total Non-Numeric Values : %d " % (total_nan))
        print("Name", "Type", "#Distinct", "NAN Values")
        columns = data.columns
        types = data.dtypes
        unique = data.nunique()
        nan_values = data.isna().sum()
        for i in range(len(data.columns)):
            print(columns[i], types[i], unique[i], nan_values[i])
    elif(isinstance(data, pyspark.sql.DataFrame)):
        data.schema
    else:
        print("Not a Dataframe" + str(type(data)))
        exit(0)

##def showImportantFeatures(model,cols): #for tree-based models
##    importances = model.feature_importances_
##    sns.barplot(x=importances, y=cols[:-1])
##    plt.show()
def showCorMatrix(df_vec):
    correlation_matrix = Correlation.corr(df_vec, "features", method="pearson").head()[0]

    print("Correlation Matrix:\n", correlation_matrix.toArray())


if __name__=="__main__":
    spambase_data = pd.read_csv('spambase.csv', header=None)

    spambase_data.columns=spambase_data.iloc[0] # copy column names
    
    data = spambase_data.drop(index=0) #remove row of columns names
    data.reset_index(drop=True, inplace=True)
    
    
    data = data.rename(columns={'Spam': 'label'})
    spark = SparkSession.builder.appName('example').getOrCreate()
    data=spark.createDataFrame(data)
    feature_list = []
    for col in data.columns:
        if col == 'label' or col == 'Label':
            continue
        else:
            feature_list.append(col)
    for col in data.columns:
        data = data.withColumn(col, data[col].cast(sql.types.FloatType()))
    #data = data.withColumnRenamed("Spam", "label")
    train_data, test_data = data.randomSplit([0.7, 0.3], seed=42)
    cv_model=None
    assembler = VectorAssembler(inputCols=feature_list, outputCol="features")
    #clf = NaiveBayes(smoothing=1.0, modelType="multinomial")
    clf = RandomForestRegressor(labelCol="label", featuresCol="features")    
    pipeline=Pipeline(stages=[assembler,clf])
    n_data=pipeline.transform(data)
    showCorMatrix(n_data)
    #clf = MultinomialNB() #Bayes
    if (not tuneHyperPar):
        cv_model = pipeline.fit(train_data)
    else:
        paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [int(x) for x in np.linspace(start = 10, stop = 50, num = 3)]) \
        .addGrid(rf.maxDepth, [int(x) for x in np.linspace(start = 5, stop = 25, num = 3)]) \
        .build()

        cross_validator = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy"),
                              numFolds=5, seed=42)
        cv_model = cross_validator.fit(train_data)
         
    preds = cv_model.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")

    accuracy = evaluator.evaluate(preds)
    print("Test set accuracy ="+str(accuracy))
    


    
