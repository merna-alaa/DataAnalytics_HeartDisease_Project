#!/usr/bin/env python
# coding: utf-8

# In[7]:


from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score,fbeta_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px


# In[10]:


df = pd.read_csv(r'C:\Users\lenovo\Downloads\heart_disease_health_indicators.csv')


# # Data cleaning 

# In[11]:


df.shape


# In[12]:


df.head()


# In[13]:


df.info()


# In[14]:


df.describe()


# In[15]:


df.isnull().sum()


# In[16]:


df.duplicated().sum()


# In[17]:


df.drop_duplicates(inplace = True)


# In[18]:


df.duplicated().sum()


# In[19]:


df.shape


# In[20]:


df['Age'] = df['Age']*4
df


# In[21]:


df['Age'].max()


# In[22]:


df['Age'].min()


# In[23]:


df["HeartDiseaseorAttack"].value_counts() 


# In[24]:


df.columns


# #Describe the dataset attributes
# 
# HeartDiseaseorAttack: A binary variable indicating whether the individual has a history of heart disease or heart attack.
# HighBP: A binary variable indicating whether the individual has high blood pressure.
# HighChol: A binary variable indicating whether the individual has high cholesterol.
# CholCheck: A binary variable indicating whether the individual has had their cholesterol checked.
# BMI: A continuous variable representing the individual's body mass index, which is a measure of body fat based on height and weight.
# Smoker: A binary variable indicating whether the individual smokes cigarettes or not.
# Stroke: A binary variable indicating whether the individual has had a stroke.
# Diabetes: A binary variable indicating whether the individual has diabetes.
# PhysActivity: A categorical variable indicating the level of physical activity of the individual.
# Fruits: A continuous variable representing the number of servings of fruits the individual consumes.
# Veggies: A continuous variable representing the number of servings of vegetables the individual consumes.
# HvyAlcoholConsump: A binary variable indicating whether the individual has heavy alcohol consumption.
# AnyHealthcare: A binary variable indicating whether the individual has any healthcare coverage.
# NoDocbcCost: A binary variable indicating whether the individual does not have any doctor's visits due to cost.
# GenHlth: A categorical variable indicating the general health status of the individual.
# MentHlth: A categorical variable indicating the mental health status of the individual.
# PhysHlth: A continuous variable representing the number of days the individual's physical health was not good.
# DiffWalk: A binary variable indicating whether the individual has difficulty walking.
# Sex: A categorical variable indicating the gender of the individual.
# Age: A continuous variable representing the age of the individual.
# Education: A categorical variable indicating the highest level of education completed by the individual.
# Income: A categorical variable indicating the income level of the individual.

# In[25]:


print("no. of column = ", len(df.columns))


# In[26]:


df1 = df.drop([ 'Education', 'Income'], axis=1)
df1


# # Data Exploration & visualization

# In[27]:


correlation = df1.corr()  
#correlation.style.background_gradient(cmap = 'BrBG')  
plt.figure(figsize = (8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',annot=True, annot_kws={'size':8}, cmap='Purples')


# In[33]:


#The interquartile range (IQR) is a measure of statistical dispersion,prints the IQR scores, which can be used to detect outliers.
Q1 = df1.quantile(0.25)
Q3 = df1.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[34]:


# Create a figure and axis
fig, ax = plt.subplots(figsize=(75, 30))

# Plot the box plots for all columns
df1.boxplot(ax=ax)
plt.ylim(bottom=0, top=2500)
# Show the plot
plt.show()


# In[35]:


sns.boxplot(df1['BMI'])


# In[36]:


sns.boxplot(df1['PhysHlth'])


# In[37]:


data = [df1["HeartDiseaseorAttack"], df1["Age"], df1["BMI"], df1["PhysHlth"]]
 
fig = plt.figure(figsize =(10, 7))
ax = fig.add_subplot(111)
 
# Creating axes instance
bp = ax.boxplot(data, patch_artist = True,
                notch ='True', vert = 0)
 
colors = ['#0000FF', '#00FF00',
          '#FFFF00', '#FF00FF']
 
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
 
# changing color and linewidth of
# whiskers
for whisker in bp['whiskers']:
    whisker.set(color ='#8B008B',
                linewidth = 1.5,
                linestyle =":")
 
# changing color and linewidth of
# caps
for cap in bp['caps']:
    cap.set(color ='#8B008B',
            linewidth = 2)
 
# changing color and linewidth of
# medians
for median in bp['medians']:
    median.set(color ='red',
               linewidth = 3)
 
# changing style of fliers
for flier in bp['fliers']:
    flier.set(marker ='D',
              color ='#e7298a',
              alpha = 0.5)
     
# x-axis labels
ax.set_yticklabels(['HeartDiseaseorAttack', 'Age',
                    'BMI', 'PhysHlth'])
 
# Adding title
plt.title("Customized box plot")
 
# Removing top axes and right axes
# ticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
     
# show plot
plt.show()


# In[38]:


df2 = df1[~((df1 < (Q1 - 1.5 * IQR)) |(df1 > (Q3 + 1.5 * IQR))).any(axis=1)]
print(df2.shape)


# In[44]:


x=['HeartDiseaseorAttack','Age','HighChol','Diabetes','HighBP', 'Fruits','Smoker']
for i in x :
    fig = px.histogram(df1, x=df1[i], nbins=50)
    fig.show()


# In[67]:


sns.pairplot(df1)


# In[47]:


sns.countplot(x='Sex', data=df1, hue='Smoker')


# In[48]:


sns.countplot(x='Smoker', data=df1, hue='HeartDiseaseorAttack')


# In[161]:


#Make a countplot on the features to differentiate them into binary, categorical and numerical features
# Countplot on each feature
plt.figure(figsize=(20,60))
for i,column in enumerate(df1.columns):
    plt.subplot(len(df1.columns), 5, i+1)
    plt.suptitle("Plot Value Count", fontsize=20, x=0.5, y=1)
    sns.countplot(data=df1, x=column)
    plt.title(f"{column}")
    plt.tight_layout()


# In[50]:


# Separate into target, binary, categorical and numerical features
target = ['HeartDiseaseorAttack']
bin_features = ['HighBP', 'HighChol', 'CholCheck','Smoker', 'Stroke','PhysActivity', 'Fruits', 'Veggies', 
                'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex']
cat_features = ['Diabetes', 'GenHlth']
num_features = ['BMI','MentHlth', 'PhysHlth', 'Age']


# In[51]:


# Explore distribtuion of binary features with pie charts
plt.figure(figsize=(20,60))
for i,column in enumerate(bin_features):
        plt.subplot(len(bin_features), 5, i+1)
        plt.pie(x=df1[column].value_counts(), labels=df1[column].unique(), autopct='%.0f%%')
        plt.title(f"{column}")
        plt.tight_layout()


# In[52]:


# Explore distribtuion of numerical features histograms
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
sns.histplot(ax=axes[0,0], data=df1, x=num_features[0])
sns.histplot(ax=axes[0,1], data=df1, x=num_features[1])
sns.histplot(ax=axes[1,0], data=df1, x=num_features[2])
sns.histplot(ax=axes[1,1], data=df1, x=num_features[3])


# In[166]:


sns.distplot(df1['Age'], kde=False, color='purple')


# In[115]:


sns.distplot(df2['BMI'], kde=False)


# In[118]:


#the relation between Body mass and heart Diseases and if body mass can affect on person with making
#it possible to have heart Diseases or any other Diseases or not
sns.countplot(x='BMI', data=df1, hue='HeartDiseaseorAttack')


# In[59]:


sns.countplot(x='BMI', data=df1, hue='Diabetes')


# In[120]:


#how healthcare is so important and how it affect on person and make him good more than people who don't have
sns.countplot(x='AnyHealthcare', data=df1, hue='HeartDiseaseorAttack')


# In[121]:


sns.countplot(x='AnyHealthcare', data=df1, hue='Diabetes')


# # Model building & Normalization

# In[60]:


x=df1.drop(['HeartDiseaseorAttack'], axis=1)
y=df1['HeartDiseaseorAttack']


# In[61]:


x_train, x_test, y_train, y_test =  train_test_split( x , y ,test_size=0.3)
y_train.value_counts()


# In[68]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# In[69]:


models={
    'LR':LogisticRegression(),
    'KNN':KNeighborsClassifier(),
    'DT':DecisionTreeClassifier(),
    'SVC':SVC(),
    'NB':GaussianNB(),
    'RF':RandomForestClassifier()
    
}


# In[70]:


for name,model in  models.items():
    print(f'using {name}: ')
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print(f'Training Accuracy :{accuracy_score(y_train,model.predict(x_train))}')
    print(f'Testing Accuracy :{accuracy_score(y_test,y_pred)}')
    print(f'Confusion matrix:\n {confusion_matrix(y_test,y_pred)}')
    print(f'Recall: {recall_score(y_test,y_pred)}')
    print(f'precision: {precision_score(y_test,y_pred)}')
    print(f'F1-score: {f1_score(y_test,y_pred)}')
    print(f'Fbeta-score: {fbeta_score(y_test,y_pred,beta=0.5)}')
    print(classification_report(y_test,y_pred))
    print('-'*33)


# Histograms: To visualize the distribution of continuous features such as age.
# Box plots: To visualize the distribution and spread of continuous features and detect outliers.
# Bar plots: To visualize the frequency distribution of categorical features such as education. 
# Heatmaps: To visualize the correlation matrix between different features.
# Scatter plots: To observe relationships between variables and uses dots to represent the relationship between them.
# countplot: To method is used to Show the counts of observations in each categorical bin using bars.
