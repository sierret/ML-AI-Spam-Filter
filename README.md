# ML-AI-Spam-Filter
An machine learning-based spam filter, Uses the UCI Spambase data(most recent version uses csv rather than uci repo package). There are two versions, one heavily uses scikit-learn and the other pyspark. Both use pandas for initial dataframe manipulation in a effort for some data cleaning consistency. Files are "preloaded" to run Multimodal Bayes and by default, RF Classification. Hypertuning [currently] oriented for RF Classification.

As such, requires packages of either:
[1]: matplotlib, seaborn,scikit-learn,pandas,numpy
[2]: pyspark,pandas,numpy

Prints Correlation Matrix of Input
Prints Feature Importance of Model
