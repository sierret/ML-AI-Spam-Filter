#from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

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
        
    else:
        print("Not a Dataframe" + str(type(data)))
        exit(0)

def showImportantFeatures(model,cols): #for tree-based models
    if (not hasattr(model,'feature_importances_')):
        return
    importances = model.feature_importances_
    sns.barplot(x=importances, y=cols[:-1])
    plt.show()
def showCorMatrix(spamData):
    spamData=spamData.iloc[:, : -1]
    correlation_matrix = spamData.corr()
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
    plt.show()

try:
    # Import for Python pre 3.6
    from idlelib.PyShell import main
except ModuleNotFoundError:
    # Import for Python version 3.6 and later
    from idlelib.pyshell import main

if __name__=="__main__":
    spambase_data = pd.read_csv('spambase.csv', header=None)

    spambase_data.columns=spambase_data.iloc[0] # copy column names
    
    data = spambase_data.drop(index=0) #remove row of columns names
    data.reset_index(drop=True, inplace=True)
    
    showCorMatrix(data)
    

    X = data.iloc[:, : -1].values #separate first n-1 column data which are features

    y = data.iloc[:,  -1].values  #separate last n column data is which is a target column

    

    X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3, random_state=99)

    clf=None
    
    if (not tuneHyperPar):
        #clf = MultinomialNB() #Bayes
        clf = RandomForestClassifier(random_state=42) #Random Forest
        #clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        clf.fit(X_train, y_train)
        showImportantFeatures(clf,spambase_data.columns)
    else:
        from sklearn.model_selection import GridSearchCV
     
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
        clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
        clf.fit(X_train, y_train)

##    pred=clf.predict(X_test)
##    print(pred)
    print("Accuracy:"+(str)(clf.score(X_test, y_test)))
    


    
