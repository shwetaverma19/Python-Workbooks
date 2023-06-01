# Titanic: Machine Learning from Disaster

The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.



## Goal

In this assignment, you will be asked to find insights from the data by using **pandas** to analyze and manipulate the data and **matplotlib** and **seaborn** for data visualization. You will get a bonus point if you can apply a logistic regression model to predict which passengers are more likely to survive in a separate test set. 


```python
# importing libraries
import os
import io
import warnings

import numpy as np
import scipy as sp
import pandas as pd
import sklearn as sk

import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

warnings.simplefilter(action='ignore', category=FutureWarning)
```


```python
# load data
titanic = pd.read_csv('https://raw.githubusercontent.com/zariable/data/master/titanic_train.csv')
display(titanic.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>


## Description of the data set
Here's a brief description of each column in the data.

- PassengerID: A column added by Kaggle to identify each row and make submissions easier
- Survived: Whether the passenger survived or not and the value we are predicting (0=No, 1=Yes)
- Pclass: The class of the ticket the passenger purchased (1=1st, 2=2nd, 3=3rd)
- Sex: The passenger's sex
- Age: The passenger's age in years
- SibSp: The number of siblings or spouses the passenger had aboard the Titanic
- Parch: The number of parents or children the passenger had aboard the Titanic
- Ticket: The passenger's ticket number
- Fare: The fare the passenger paid
- Cabin: The passenger's cabin number
- Embarked: The port where the passenger embarked (C=Cherbourg, Q=Queenstown, S=Southampton)

### Find the number of missing values for each column.**
The first step in data analysis is to identify columns with missing data. Can you find the columns in this data with missing value as well as the number of records with missing value for each column?  

Hint: you will need [isna](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.isna.html) function.


```python

missing_columns = titanic.columns[titanic.isna().any()].tolist()
missing_values = titanic[missing_columns].isna().sum()
# print the missing columns and the number of missing values 
print("Missing_columns",missing_columns)
print("Total Missing_values:", )
print("Missing_values:")
print(missing_values)
```

    Missing_columns ['Age', 'Cabin', 'Embarked']
    Total Missing_values:
    Missing_values:
    Age         177
    Cabin       687
    Embarked      2
    dtype: int64


### **Impute missing values.**
Now we've identified the following columns with missing values: _Age_, _Cabin_ and _Embarked_. As the next step, we want to impute those missing values. There are three ways to impute the missing values:
- A constant value that has meaning within the domain.
- The mean, median or mode value based on non-missing values of that column.
- A random value drawn from other non-missing values of that column.


- the missing values of column _age_ with the mean of that column.
- the missing values of column _Cabin_ with a constant value 'other'.
- the missing values of column _Embarked_ with the [mode](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.mode.html) of that column.


```python

mean_age = titanic['Age'].mean()
titanic_updated = titanic.assign(Age=titanic.Age.fillna(mean_age))


titanic_updated = titanic_updated.assign(Cabin=titanic.Cabin.fillna('other'))


mode_embarked = titanic_updated['Embarked'].mode()[0]
titanic_updated = titanic_updated.assign(Embarked=titanic_updated.Embarked.fillna(mode_embarked))

```

### ** What's the name of the person who has the 16th most expensive ticket?**


```python

titanic_updated
sort_Fare=titanic_updated.sort_values(by=['Fare'],ascending=False)
#sort_Fare.head(20)
sort_Fare.iloc[15]["Name"]


```




    'Farthing, Mr. John'



### **Out of all the remales who survived, what's the name who has the 6th most expensive ticket?**


```python

filtered_titanic=titanic_updated.loc[ (titanic_updated['Sex']=='female') & (titanic_updated['Survived']==1) ]
filtered_titanic=filtered_titanic.sort_values(by=['Fare'],ascending=False)
filtered_titanic.head(10)
filtered_titanic.iloc[5]["Name"]

```




    'Baxter, Mrs. James (Helene DeLaudeniere Chaput)'



### **Examine the survival rate**
Calculate the survival rate for different gender and Pclass combination and use a couple of sentences to describe your findings. Hint: pivot_table is your friend.


```python

titanic_updated.pivot_table('Survived', index='Sex', columns='Pclass')

'''
Findings:
According to the pivot table, female passengers have a much higher survival rate than male passengers in all Pclasses.
Pclass1 has the highest survival rate for both male and female passengers.
Females generally have higher chances of survival as compared to men irrespective of any ticket purchased.
Moreover, womens with 1st and 2nd class ticket have higher chances for survival as compared to
a women with a 3rd class ticket. Furthermore, mens with 1st class ticket have higher chances of 
survival followed by 2nd and 3rd class ticket.  '''

```




    '\nFindings:\nAccording to the pivot table, female passengers have a much higher survival rate than male passengers in all Pclasses.\nPclass1 has the highest survival rate for both male and female passengers.\nFemales generally have higher chances of survival as compared to men irrespective of any ticket purchased.\nMoreover, womens with 1st and 2nd class ticket have higher chances for survival as compared to\na women with a 3rd class ticket. Furthermore, mens with 1st class ticket have higher chances of \nsurvival followed by 2nd and 3rd class ticket.  '



### **Is Age or Fare an important factor to one's chance of survival?**
Visualize the distribution of Column _Age_ for both survived and non-survived population and write down your findings based on the visualization.


```python

sns.scatterplot(data=titanic_updated, x="Fare", y="Age", hue="Survived")

```




    <AxesSubplot:xlabel='Fare', ylabel='Age'>




    
![png](output_16_1.png)
    



```python
'''
#Findings:
Fare is an important feature and it is likely that passengers who paid higher fares 
had a better chance of survival. This can be seen by comparing the survival rates of various fare groups. 
Passengers who paid higher fares, for example, had a better chance of survival, 
as we can see as the fare starts increasing there are very more survival.
Age too is an next important factor; children under the age of 10-15 had a higher chance of survival 
than other age groups.'''
```




    '\n#Findings:\nFare is an important feature and it is likely that passengers who paid higher fares \nhad a better chance of survival. This can be seen by comparing the survival rates of various fare groups. \nPassengers who paid higher fares, for example, had a better chance of survival, \nas we can see as the fare starts increasing there are very more survival.\nAge too is an next important factor; children under the age of 10-15 had a higher chance of survival \nthan other age groups.'



### ** Calculate and visualize the survival rate for discrete columns**
- Calculate the survival rate for column _SibSp_ and _Parch_.
- Use sns.barplot to visualize the survival rate for column _SibSp_ and _Parch_.


```python
sibsp_survival = titanic_updated.groupby("SibSp")["Survived"].mean()
print(sibsp_survival)

# Calculating the survival rate for Parch
parch_survival = titanic_updated.groupby("Parch")["Survived"].mean()
print(parch_survival)
```

    SibSp
    0    0.345395
    1    0.535885
    2    0.464286
    3    0.250000
    4    0.166667
    5    0.000000
    8    0.000000
    Name: Survived, dtype: float64
    Parch
    0    0.343658
    1    0.550847
    2    0.500000
    3    0.600000
    4    0.000000
    5    0.200000
    6    0.000000
    Name: Survived, dtype: float64



```python

sns.barplot(data=titanic_updated, x="Parch", y="Survived")

```




    <AxesSubplot:xlabel='Parch', ylabel='Survived'>




    
![png](output_20_1.png)
    



```python
sns.barplot(data=titanic_updated, x="SibSp", y="Survived")

```




    <AxesSubplot:xlabel='SibSp', ylabel='Survived'>




    
![png](output_21_1.png)
    


### Find the correlations.**
Find the correlations between the feature and the target variable _Survived_ and use heatmap to visualize it. Summarize your findings.


```python
corr = titanic_updated.corr()
#sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
sns.heatmap(corr, annot=True)
plt.show()
```


    
![png](output_23_0.png)
    



```python

# Select features and target
features = titanic_updated.drop(columns=["Survived", "PassengerId"])
target = titanic_updated["Survived"]

# Calculate the correlation matrix
corr = features.corrwith(target)

sns.heatmap(corr.to_frame(),annot=True)
plt.title('Heatmap of Correlation of features with Survived')
plt.ylabel('Features')
plt.xlabel('Survived')

plt.show()
```


    
![png](output_24_0.png)
    



```python
'''
#Findings:
The heatmap shows that the feature 'Pclass' has the highest negative correlation with the target variable 'Survived'(-0.33) 
and the feature 'Fare' has the highest positive correlation with 'Survived' (0.25). 
It means that the chances of survival are higher for passengers who paid a higher fare and 
lower for passengers who paid a lower fare.
It's also worth noting that other characteristics like 'Sex' and 'Embarked' 
have a moderate correlation with the target variable 'Survived'. 
In summary, the features 'Pclass,' 'Fare,' 'Sex,' and 'Embarked' in the Titanic dataset 
are moderately correlated with the target variable 'Survived,' with 'Fare' having the highest positive correlation
and 'Pclass' having the highest negative correlation'''
```




    "\n#Findings:\nThe heatmap shows that the feature 'Pclass' has the highest negative correlation with the target variable 'Survived'(-0.33) \nand the feature 'Fare' has the highest positive correlation with 'Survived' (0.25). \nIt means that the chances of survival are higher for passengers who paid a higher fare and \nlower for passengers who paid a lower fare.\nIt's also worth noting that other characteristics like 'Sex' and 'Embarked' \nhave a moderate correlation with the target variable 'Survived'. \nIn summary, the features 'Pclass,' 'Fare,' 'Sex,' and 'Embarked' in the Titanic dataset \nare moderately correlated with the target variable 'Survived,' with 'Fare' having the highest positive correlation\nand 'Pclass' having the highest negative correlation"



### ** Any other insights do you draw by analyzing the data? Summarize the findings as well as provide the code leading you to the findings.**


```python

```


```python

sns.barplot(data=titanic_updated, x="Pclass", y="Survived")
```




    <AxesSubplot:xlabel='Pclass', ylabel='Survived'>




    
![png](output_28_1.png)
    



```python
sns.barplot(data=titanic_updated, x="Parch", y="Survived")
```




    <AxesSubplot:xlabel='Parch', ylabel='Survived'>




    
![png](output_29_1.png)
    



```python
sns.barplot(data=titanic_updated, x="SibSp", y="Survived")
```




    <AxesSubplot:xlabel='SibSp', ylabel='Survived'>




    
![png](output_30_1.png)
    



```python
sns.barplot(x='Pclass', y='Age', hue='Survived',hue_order=[0, 1], data=titanic_updated, ci=None, )
plt.legend(labels=["No", "Yes"])
```




    <matplotlib.legend.Legend at 0x7fe1c706f070>




    
![png](output_31_1.png)
    



```python
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic_updated, ci=None)
```




    <AxesSubplot:xlabel='Pclass', ylabel='Survived'>




    
![png](output_32_1.png)
    



```python
'''
#Findings:
As we can see from the above graph, females coming from any Pclass had higher chances of survival as compared to mens.
People with high Pclass(1st) were higher edge to survive as compared to other Pclass(2nd and 3rd)
People with one sibling/spouse and three Parch have higher chances of survival. 
It is evident from the graph above that children, women, higher Pclass people had the higher 
chances of survival as compared to other'''

```




    '\n#Findings:\nAs we can see from the above graph, females coming from any Pclass had higher chances of survival as compared to mens.\nPeople with high Pclass(1st) were higher edge to survive as compared to other Pclass(2nd and 3rd)\nPeople with one sibling/spouse and three Parch have higher chances of survival. \nIt is evident from the graph above that children, women, higher Pclass people had the higher \nchances of survival as compared to other'



### **Build a ML model to predict survival.**
Building a logistic regression model to predict the probability of survival for all the passengers in this [file](https://raw.githubusercontent.com/zariable/data/master/titanic_test.csv)? Evaluating your model accuracy on [Kaggle](https://www.kaggle.com/c/titanic). 


```python

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = pd.read_csv('https://raw.githubusercontent.com/zariable/data/master/titanic_test.csv')

features = titanic_updated.drop(columns=["Survived", "PassengerId"])
target = titanic_updated["Survived"]

features = pd.get_dummies(features)


# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=90)

# print(X_train)

#passenger_id = X_test["PassengerId"]
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train, y_train)



# test_data = pd.read_csv("/Users/himeshk/Desktop/_COLLEGE/WINTER QUARTER/522(Advanced data mining)-Yu/Assi1/kaggle_dataset.csv")

# features = test_data.drop(columns=["PassengerId"])
# target = test_data["Survived"]
# # print(len(features), len(X_train))
# test_X_train, test_X_test, test_Y_train, test_Y_test = train_test_split(features, target, test_size=0, random_state=90)
# print(X_train)
# print(test_X_train)
# PassengerId_test= test_data["PassengerId"]

# y_pred = model.predict(features)




```

    /Users/himeshk/opt/anaconda3/lib/python3.9/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(





<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div>




```python
'''Several things can be done to improve the model's performance:

1.Tuning the model's hyperparameters: You can change the model's parameters to improve its performance.
2.Feature engineering: we can have new features from existing data to provide more information to the model.
3.Ensemble methods: these actually combine multiple of the models to produce a more robust and accurate prediction.
4.Using another algorithm: we can compare the performance of other algorithms such as Random Forest, SVM, Neural Network, logistic regression.
5.Adding external data: You can include external data'''
```




    "Several things can be done to improve the model's performance:\n\n1.Tuning the model's hyperparameters: You can change the model's parameters to improve its performance.\n2.Feature engineering: we can have new features from existing data to provide more information to the model.\n3.Ensemble methods: these actually combine multiple of the models to produce a more robust and accurate prediction.\n4.Using another algorithm: we can compare the performance of other algorithms such as Random Forest, SVM, Neural Network, logistic regression.\n5.Adding external data: You can include external data"




```python

```
