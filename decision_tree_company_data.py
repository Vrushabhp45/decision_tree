import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder #for encoding
from sklearn.model_selection import train_test_split #for train test splitting
from sklearn.tree import DecisionTreeClassifier #for decision tree object
from sklearn.metrics import classification_report, confusion_matrix #for checking testing results
from sklearn.tree import plot_tree #for visualizing tree
#reading the data
df = pd.read_csv('C:/Users/HP/PycharmProjects/Excelrdatascience/Company_Data.csv')
df.head()
#getting information of dataset
df.info()
df.shape
df.isnull().any()
# let's plot pair plot to visualise the attributes all at once
sns.pairplot(data=df, hue = 'ShelveLoc')
# Converting categorical into numerical
#get all categorical columns
cat_columns = df.select_dtypes(['object']).columns

#convert all categorical columns to numeric
df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
df.head()
df.info()
#Declare feature vector and target variable
x=df.iloc[:,0:6]
y=df['ShelveLoc']
x.head()
y.head()
df['ShelveLoc'].unique()
df.ShelveLoc.value_counts()
colnames = list(df.columns)
colnames
#Splitting data into training and testing data set
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)
#Building Decision Tree Classifier using Entropy Criteria
model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)
#Plot Decision Tree
#PLot the decision tree
from sklearn import tree
fig = plt.figure(figsize=(25,20))
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model);
#Predict the results
y_pred = model.predict(x_test)

y_pred
#Check accuracy score
from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
#Compare the train-set and test-set accuracy
y_pred_train = model.predict(x_train)

y_pred_train
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
#Confusion matrix
# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])
# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm)

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
#Classification metrices
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
