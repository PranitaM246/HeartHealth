import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings

df = pd.read_csv('C:/Users/Ammar/heart.csv')
df.head()
print(df.isnull().sum())
df = df.dropna()
df.info()
df.describe()

import pandas_profiling
pandas_profiling.ProfileReport(df)

d = df['Target'].value_counts()
print(d)

import seaborn as sns

def plotTarget():
    sns.countplot(x='Target', data=df, ax=ax)
    for i, p in enumerate(ax.patches):
        count = df['Target'].value_counts().values[i]
        x = p.get_x() + p.get_width() / 2.
        y = p.get_height() + 2
        label = '{:1.2f}'.format(count / float(df.shape[0]))
        ax.text(x, y, label, ha='center')

fig_target, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 2))
plotTarget()

df.corr()

def plotAge():
    facet_grid = sns.FacetGrid(df, hue='Target')
    facet_grid.map(sns.kdeplot, "Age", fill=True, ax=axes[0])
    legend_labels = ['Disease false', 'Disease true']
    for t, l in zip(axes[0].get_legend().texts, legend_labels[::-1]):
        t.set_text(l)
        axes[0].set(xlabel='Age', ylabel='Density')

    avg = df[["Age", "Target"]].groupby(['Age'], as_index=False).mean()
    sns.barplot(x='Age', y='Target', data=avg, ax=axes[1])
    axes[1].set(xlabel='Age', ylabel='disease probability')

    plt.clf()

fig_age, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 8))
plotAge()

x = df['ChestPain']
x.value_counts()
x = df['Thal']
x.value_counts()

import seaborn as sns

category = [('ChestPain', ['typical', 'nontypical', 'nonanginal', 'asymptomatic']),
            ('Thal', ['fixed', 'normal', 'reversable'])]

continuous = [('Age', 'Age in year'),
              ('Sex', '1 for Male 0 for Female'),
              ('RestBP', 'BP in Rest State'),
              ('Fbs', 'Fasting blood glucose'),
              ('RestECG', 'ECG at rest'),
              ('Chol', 'serum cholestoral in mg/d'),
              ('MaxHR', 'Max Heart Rate'),
              ('ExAng', 'Exchange Rate'),
              ('Slope', 'Slope of Curve'),
              ('Oldpeak', 'ST depression by exercise relative to rest'),
              ('Ca', '# major vessels: (0-3) colored by flourosopy')]

def plotCategorial(attribute, labels, ax_index):
    sns.countplot(x=attribute, data=df, ax=axes[ax_index][0])
    sns.countplot(x='Target', hue=attribute, data=df, ax=axes[ax_index][1])
    avg = df[[attribute, 'Target']].groupby([attribute], as_index=False).mean()
    sns.barplot(x=attribute, y='Target', data=avg, ax=axes[ax_index][2])
    for t, l in zip(axes[ax_index][1].get_legend().texts, labels):
        t.set_text(l)
    for t, l in zip(axes[ax_index][2].get_legend().texts, labels):
        t.set_text(l)

def plotContinuous(attribute, xlabel, ax_index):
    sns.histplot(df[[attribute]], ax=axes[ax_index][0], stat="density")
    axes[ax_index][0].set(xlabel=xlabel, ylabel='density')
    sns.violinplot(x='Target', y=attribute, data=df, ax=axes[ax_index][1])

def plotGrid(isCategorial):
    if isCategorial:
        [plotCategorial(x[0], x[1], i) for i, x in enumerate(category)]
    else:
        [plotContinuous(x[0], x[1], i) for i, x in enumerate(continuous)]

fig_categorial, axes = plt.subplots(nrows=len(category), ncols=3, figsize=(10, 10))
plotGrid(isCategorial=True)

fig_continuous, axes = plt.subplots(nrows=len(continuous), ncols=2, figsize=(10, 10))
plotGrid(isCategorial=False)

chestpain_dummy = pd.get_dummies(df['ChestPain'])
chestpain_dummy.rename(columns={1: 'Typical', 2: 'Asymptomatic', 3: 'Nonanginal', 4: 'Nontypical'}, inplace=True)

restecg_dummy = pd.get_dummies(df['RestECG'])
restecg_dummy.rename(columns={0: 'Normal_restECG', 1: 'Wave_abnormal_restECG', 2: 'Ventricular_ht_restECG'}, inplace=True)

slope_dummy = pd.get_dummies(df['Slope'])
slope_dummy.rename(columns={1: 'Slope_upsloping', 2: 'Slope_flat', 3: 'Slope_downsloping'}, inplace=True)

thal_dummy = pd.get_dummies(df['Thal'])
thal_dummy.rename(columns={3: 'Thal_Normal', 6: 'Thal_fixed', 7: 'Thal_reversible'}, inplace=True)

df = pd.concat([df, chestpain_dummy, restecg_dummy, slope_dummy, thal_dummy], axis=1)
df.drop(['ChestPain', 'RestECG', 'Slope', 'Thal'], axis=1, inplace=True)

df.info()
df.head()

df_X = df.loc[:, df.columns != 'Target']
df_y = df.loc[:, df.columns == 'Target']

import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

selected_features = []
lr = LogisticRegression(max_iter=50)
rfe = RFE(lr, 20)

warnings.simplefilter('ignore')
rfe.fit(df_X.values, df_y.values)
print(rfe.support_)
print(rfe.ranking_)

for i, feature in enumerate(df_X.columns.values):
    if rfe.support_[i]:
        selected_features.append(feature)

df_selected_X = df_X[selected_features]
df_selected_y = df_y

lm = sm.Logit(df_selected_y, df_selected_X)
result = lm.fit()

print(result.summary2())

warnings.simplefilter('ignore')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_selected_X, df_selected_y, test_size=0.25, random_state=10)
columns = X_train.columns

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def cal_accuracy(y_test, y_predict): 
    print("\nConfusion Matrix: \n", confusion_matrix(y_test, y_predict)) 
    print(f"\nAccuracy : {accuracy_score(y_test, y_predict)*100:0.3f}")

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_predict = lr.predict(X_test)

print(f"Accuracy of Test Dataset: {lr.score(X_test, y_test):0.3f}")
print(f"Accuracy of Train Dataset: {lr.score(X_train, y_train):0.3f}")

warnings.simplefilter('ignore')
print("Predicted values:") 
print(y_predict)
cal_accuracy(y_test, y_predict)

from sklearn import svm

svm_linear = svm.SVC(kernel='linear')
svm_linear.fit(X_train, y_train)

warnings.simplefilter('ignore')
print(f"Accuracy of Test Dataset: {svm_linear.score(X_test, y_test):0.3f}")
print(f"Accuracy of Train Dataset: {svm_linear.score(X_train, y_train):0.3f}")

print("Predicted values:") 
print(y_predict)
cal_accuracy(y_test, y_predict)

from sklearn.tree import DecisionTreeClassifier

gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=2, min_samples_leaf=5)
gini.fit(X_train, y_train)

warnings.simplefilter('ignore')
print(f"Accuracy of Test Dataset: {gini.score(X_test, y_test):0.3f}")
print(f"Accuracy of Train Dataset: {gini.score(X_train, y_train):0.3f}")

y_predict = gini.predict(X_test) 
print("Predicted values:\n")
print(y_predict) 
cal_accuracy(y_test, y_predict)

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=50)
forest.fit(X_train, y_train)

warnings.simplefilter('ignore')
print(f"Accuracy of Test Dataset: {forest.score(X_test, y_test):0.3f}")
print(f"Accuracy of Train Dataset: {forest.score(X_train, y_train):0.3f}")

y_predict = forest.predict(X_test)
print("Predicted values:\n")
print(y_predict)
cal_accuracy(y_test, y_predict)

from sklearn import model_selection

kfold = model_selection.KFold(n_splits=8, random_state=7, shuffle=True)
models = [('Linear Regression', lr), ('Support Vector Machine', svm_linear),
          ('Decision Tree', gini), ('Random Forest', forest)]

warnings.simplefilter('ignore')

for model in models:
    results = model_selection.cross_val_score(model[1], X_train, y_train, cv=kfold, scoring='accuracy')
    print(f"Cross validated Accuracy of {model[0]}:: {results.mean():.3f}")

models = pd.DataFrame({
    'Model': ['Logistics Regression', 'SVM', 'Decision Tree', 'Random Forest'],
    'Traning Accuracy': [lr.score(X_train, y_train), svm_linear.score(X_train, y_train),
                         gini.score(X_train, y_train), forest.score(X_train, y_train)],
    'Test Accuracy': [lr.score(X_test, y_test), svm_linear.score(X_test, y_test),
                     gini.score(X_test, y_test), forest.score(X_test, y_test)]
})
models.sort_values(by='Test Accuracy', ascending=False)
