
<a href="http://www.calstatela.edu/centers/hipic"><img align="left" src="https://avatars2.githubusercontent.com/u/4156894?v=3&s=100"><image/>
</a>
<img align="right" alt="California State University, Los Angeles" src="http://www.calstatela.edu/sites/default/files/groups/California%20State%20University%2C%20Los%20Angeles/master_logo_full_color_horizontal_centered.svg" style="width: 360px;"/>

# CIS5560 Term Project Tutorial


#### Authors:Hemamalini Madhanguru,Lakshmi Sundararajan,Pallavi Attimakula

#### Instructor: [Jongwook Woo](https://www.linkedin.com/in/jongwook-woo-7081a85)

#### Date: 05/18/2017


### Cluster creation and specification

Click on the cluster tab on the left pane and specify a name for the cluster and click on Create Cluster.

These are the configuration options for the cluster, 
Spark Version : Spark 2.1 (Auto-updating, Scala 2.10) 
Memory – 6GB Memory , 0.88 Cores, 1 DBU 
File System – DBFS (Data Bricks File System)


### Prepare the Data

First, import the dataset manually using the tables table in the left pane to upload the data, upon uploading the data give the table a name and select the apporpriate datatype for the data.

#### Creating a Regression Model
In this exercise, you will implement a regression model using Linear Regression that uses features of lending loan clubs to predict the value of the installments.

You should follow the steps below to build, train and test the model from the source data:
1. Build a schema of a source data for its Data Frame
2. Load the Source Data to the schema
3. Prepare the data with the features (input columns, output column as label)
4. Split the data using data.randomSplit(): Training and Testing
5. Transform the columns to a vector using VectorAssembler
6. set features and label from the vector
7. Build a LinearRegression Model with the label and features
8. Train the model
9. Prepare the testing Data Frame with features and label from the vector; Rename label to trueLabel
10. Predict and test the testing Data Frame using the model trained at the step 8
11. Compare the predicted result and trueLabel

#### Import Spark SQL and Spark ML Libraries
First, import the libraries you will need:


```python
# Import Spark SQL and Spark ML libraries
from pyspark.sql.types import *
from pyspark.sql.functions import *

from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression , DecisionTreeRegressor
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator
```

#### Prepare the Data
Most modeling begins with exhaustive exploration and preparation of the data. In this example, you will simply select a subset of columns to use as features as well as the ArrDelay column, which will be the label your model will predict.


```python
data = sqlContext.sql("SELECT ACTIVITY,NAME, CAST(count(VIOLATION_CODE) AS DOUBLE) as Total_violations,grade, CAST(score as DOUBLE) as label, CAST(sum(points) as DOUBLE) as Violation_points FROM rest_vio where score >= '65' group by NAME,ACTIVITY,grade,score")
data.show()
```

#### Split the Data
It is common practice when building supervised machine learning models to split the source data, using some of it to train the model and reserving some to test the trained model. In this exercise, you will use 70% of the data for training, and reserve 30% for testing.


```python
#Feature selection
data_fea = data.select("label","Total_violations")
#Data split to train and test samples. 70% for training and 30% for testing
splits = data_fea.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")
print "We have %d training examples and %d test examples." % (train.count(), test.count())
```

### Prepare the Training Data
To train the regression model, you need a training data set that includes a vector of numeric features, and a label column.


```python
#Model1 - Linear Regression
vectorAssembler = VectorAssembler(inputCols=["Total_violations"], outputCol="features")
lr = LinearRegression(labelCol="label",featuresCol="features", maxIter=10, regParam=0.3)
pipeline = Pipeline(stages=[vectorAssembler, lr])
```

### Using trainvalidationsplit
It's going to use 80% of the data that it's got in its training set to train the model and then the remaining 20% is going to use to validate the trained model.


```python
paramGrid1 = ParamGridBuilder().addGrid(lr.regParam, [0.3, 0.01]).addGrid(lr.maxIter, [10, 5]).build()
tvs = TrainValidationSplit(estimator=pipeline, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid1, trainRatio=0.8)
model1 = tvs.fit(train)
```


    

    NameErrorTraceback (most recent call last)

    <ipython-input-2-7fb0a4d52430> in <module>()
    ----> 1 paramGrid1 = ParamGridBuilder().addGrid(lr.regParam, [0.3, 0.01]).addGrid(lr.maxIter, [10, 5]).build()
          2 tvs = TrainValidationSplit(estimator=pipeline, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid1, trainRatio=0.8)
          3 model1 = tvs.fit(train)


    NameError: name 'lr' is not defined


### Test the Model
Now you're ready to use the transform method of the model to generate some predictions. 


```python
prediction1 = model1.transform(test)
# LinearRegression
predicted1 = prediction1.select("features", "prediction", "trueLabel")
display(predicted1)
```


    

    NameErrorTraceback (most recent call last)

    <ipython-input-3-f2c99e9b079b> in <module>()
    ----> 1 prediction1 = model1.transform(test)
          2 # LinearRegression
          3 predicted1 = prediction1.select("features", "prediction", "trueLabel")
          4 display(predicted1)


    NameError: name 'model1' is not defined


### RMSE
The regression line predicts the average value associated with a given a x value. To do this we use the root mean square error.


```python
# LinearRegression: predictionCol="prediction", metricName="rmse"
evaluator1 = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
rmse1 = evaluator.evaluate(prediction1)
print "Root Mean Square Error (RMSE):", rmse1
```

### Gradient Boost Regressor
Gbt is a learning algorithm for regression. It supports both continous and categorical features. 
This operation is ported from Spark ML.



```python
#Model2 - GBT regressor
vectorAssembler2 = VectorAssembler(inputCols=["Total_violations"], outputCol="features")
dt = DecisionTreeRegressor(labelCol="label", featuresCol="features")
pipeline2 = Pipeline(stages=[vectorAssembler2, dt])
```

#### Parameter Grid Builder
Builder for a param grid used in grid search-based model selection. Validation for hyper-parameter tuning splits the input dataset into train and validation sets, and uses evaluation metric on the validation set to select the best model.


```python
# Define a grid of hyperparameters to test:
#  - maxDepth: max depth of each decision tree in the GBT ensemble
#  - maxIter: iterations, i.e., number of trees in each GBT ensemble

paramGrid2 = ParamGridBuilder()\
  .addGrid(gbt.maxDepth, [2, 10])\
  .addGrid(gbt.maxIter, [10, 200])\
  .build()
# We define an evaluation metric.  This tells CrossValidator how well we are doing by comparing the true labels with predictions.
#evaluator = RegressionEvaluator(metricName="rmse", labelCol=gbt.getLabelCol(), predictionCol=gbt.getPredictionCol())
# Declare the CrossValidator, which runs model tuning for us.
#cv = CrossValidator(estimator=gbt, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid2, numFolds=10)

tvs2 = TrainValidationSplit(estimator=pipeline2, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid2, trainRatio=0.8)
model2 = tvs2.fit(train)
```


```python
prediction2 = model2.transform(test)
predicted2 = prediction2.select("features", "prediction", "trueLabel")
display(predicted2)
```


```python
rmse = evaluator.evaluate(predictions2)
print "RMSE on our test set: %g" % rmse
```

References:
1. [Markdown Cells in Jupyter](http://jupyter-notebook.readthedocs.io/en/latest/examples/Notebook/Working%20With%20Markdown%20Cells.html)
1. [Markdown Cheatshee](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)
1. [Markdown Guide](https://help.ghost.org/hc/en-us/articles/224410728-Markdown-Guide)
