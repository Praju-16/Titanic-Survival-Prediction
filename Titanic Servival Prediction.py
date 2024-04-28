#!/usr/bin/env python
# coding: utf-8

# # TITANIC SURVIVAL PREDICTION

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data=pd.read_csv('C:\\Users\\AKSHAY\\Downloads\\train.csv')
data


# In[3]:


data.info()


# In[4]:


data.isnull()


# In[5]:


data.describe()


# In[6]:


data.isnull().sum()


# In[7]:


data = data.drop(columns='Cabin', axis = 1)


# In[8]:


# replacing the missing values in age column with mean value
data['Age'].fillna(data['Age'].mean(), inplace=True)


# In[9]:


data.isnull().sum()


# In[10]:


# finding the mode value of Embarked column (mode means most no. of time value)
print(data["Embarked"].mode())


# In[11]:


print(data["Embarked"].mode()[0])


# In[12]:


data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)


# In[13]:


data.isnull().sum()


# # DATA ANALYSIS

# In[14]:


data['Survived'].value_counts()


# In[15]:


sns.countplot('Survived', data=data)


# In[16]:


sns.countplot('Sex', data=data)


# In[17]:


sns.countplot('Sex', hue='Survived',data=data)


# In[18]:


sns.countplot('Pclass', hue='Survived',data=data)


# In[19]:


sns.factorplot('Pclass',y = 'Survived',hue = 'Sex',data=data)
plt.show()


# In[20]:


# Encoading the categorical columns
data['Sex'].value_counts()


# In[21]:


data['Embarked'].value_counts()


# In[22]:


# converting categorical columns 
data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)


# In[23]:


data.head()


# In[24]:


x=data.drop(columns = ['PassengerId','Name','Ticket','Survived'], axis = 1)
y=data['Survived']


# In[25]:


print(x)


# In[26]:


print(y)


# In[27]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# In[28]:


# splitting data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2, random_state=2)


# In[29]:


print(x.shape, x_train.shape, x_test.shape)


# In[31]:


# model training
model = LogisticRegression()


# In[32]:


model.fit(x_train, y_train)


# In[33]:


x_train_prediction = model.predict(x_train)


# In[34]:


training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print('Accuracy score of training data: ',training_data_accuracy)


# In[35]:


x_test_prediction = model.predict(x_test)


# In[36]:


testing_data_accuracy = accuracy_score(y_test, x_test_prediction)
print('Accuracy score of testing data: ',testing_data_accuracy)


# In[38]:


data.corr()


# In[43]:


# The first row contains the values that represent the correlation of each variable with the target variable. 
# ‘Age’ and ‘Fare’ are highly (positively) correlated with the target variable.
colormap = plt.cm.Blues
plt.figure(figsize=(10,7))
sns.heatmap(data.corr(), cmap=colormap, annot=True, linewidths=0.1)


# In[40]:


# The enhanced box plot shown above indicates that the fare of “Female” passengers is on average higher than male passengers. 
# It could be because of the additional services offered to female passengers.
sns.catplot(x='Sex', y='Fare', data=data, kind='boxen')


# In[41]:


# We infer that most of the older people were traveling in first class. 
# It may be because they were rich. The youngsters who are aged between 25 and 35 were mostly traveling in second and third classes.
sns.catplot(x='Sex', y='Age', data=data, kind='box', hue='Pclass')


# In[42]:


plt.figure(figsize=(8, 5))
sns.histplot(data=data, x='Age', bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[ ]:




