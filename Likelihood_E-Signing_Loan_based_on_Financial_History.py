#!/usr/bin/env python
# coding: utf-8

# # Likelihood of E-Signing a Loan based on Financial History
# - We are going to asses the 'quality' of leads our company receives from market place regarding the loan request.
# - We'll predict the likelihood of the customer actually going through the whole process of completing the E-signing process.
# 
# ## Business Challenge
# 
# - Here we are working for Fintech company.
# - We are given the task of whether or not to let the customer complete the onbaording E-signing process(e_signed).
# - The funnel of the whole process is as follows - The customers will come to the company portal and provide their information
# - After deciding to accept the lead as to whether the customer will complete the whole process of compeleting till the E-sgining screen.
# - Our job is to predict that the lead is a 'quality' one or not i.e whether what is the likelihood of the applicant to actually complete the whole process.
# 
# ## DATA
# - The data we are going to recive is not going to be the raw data from the applicant.
# - We will recive the data that is been passed through an algorithm that genreates risk scores on the basis of the information provided by the applicant.
# - Our job is to leverage these risk scores and give our predictions on the likelihood of the applicant to finsih till the E-signing process.
# - We have the following columns in our dataset:
#     - entry_id : That is the entry ID for our applicant which is unique for every applicant.
#     - age : This will be the age of our applicant.
#     - pay_schedule : This is how often how our applicant is paid.
#     - home_owner : Whether the user owns a house 0 or 1.
#     - income : monthly income of our applicant.
#     - years_employed : years the applicant is employed.
#     - current_address_year : Number of years the applicant has stayed at their current address.
#     - personel_account_m : Number of months the person had the account it is correlated with nes column.
#     - personel_account_y : Number of years the person had the account i.e  months 2 & years 3 then person has the account for 3 years and 2 months.
#     - has_debt : If the applicant has a prior dent 0 or 1.
#     - risk_score, _2,_3,_4,_5: Predict whether the user will complete the whole process of E-Signing the loan and Payback which are based on different factors.
#     - ext_quality_score, _2 : The are external quality score based on factors.
#     - inquiries_last_month : how many inquiries user has in last month.
#     - e-signed : If the applicant e-signed the lan or not. 


import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# Cleaning our data a bit before actually applying any models on them
dataset = pd.read_csv('Financial-Data.csv')
dataset.head()



dataset.isna().any()



# Now we'll be creating a Histograms for our given dataset to find insights

dataset2 = dataset.drop(columns= ['entry_id', 'pay_schedule', 'e_signed'])




fig = plt.figure(figsize=(20,48))
plt.title("Histograms for our Dataset", fontsize= 20)
for i in range(dataset2.shape[1]):
    plt.subplot(14,2,i+1)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i])
    
    vals = np.size(dataset2.iloc[:, i].unique())
    if vals >= 100:
        vals = 100
    plt.hist(dataset2.iloc[:,i], bins = vals, color = "#3f5d7d")
plt.subplots_adjust(hspace= 0.6)


# - Now to take the insights from the histograms
# - We'll move forward with finding the correalation between every attribute in our dataset to our response attribute.



dataset2.corrwith(dataset.e_signed).plot.bar(figsize = (20,10), 
                                             title = 'Correlation with E-Signed', 
                                             fontsize =15, 
                                             rot = 80,
                                             grid = True)


# - Building a correlation matrix to to check how the relation of the attribute with each other



corr = dataset2.corr()

# Generate the mask for the upper triangle
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

# Setup the matplotlib figure
f, ax = plt.subplots(figsize=(18,15))

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap= True)

# Draw the heatmap with mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3 , center=0, square=True, linewidths=0.5, cbar_kws={"shrink" : 0.5})




import random
random.seed(100)




# We removed the individual months and years and converted it to single attribute of total months
dataset = dataset.drop(columns=['months_employed'])
dataset['personal_account_months'] = (dataset.personal_account_m + (dataset.personal_account_y * 12))
dataset[['personal_account_m','personal_account_y', 'personal_account_months']].head()




dataset = dataset.drop(columns=['personal_account_m','personal_account_y'])




# Now we will preprocess our data with One Hot Encoding and getting our dummy variables
dataset = pd.get_dummies(dataset)
dataset.columns




dataset = dataset.drop(columns = ['pay_schedule_semi-monthly'])




# Removing Extra columns for train_test_split
response = dataset['e_signed']
dataset = dataset.drop(columns=['e_signed'])




# Splitting dataset into test set and train set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(dataset, response, test_size = 0.2, random_state = 0)




#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2




from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state=0, n_estimators=100, criterion='entropy')
classifier.fit(X_train,y_train)




# Predicting the test set results
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))




# Applying Parameter tuning using Grid Search
import time
parameters = {'max_depth' : [3, None],
             'max_features' : [1,5,10],
             'min_samples_split' : [2,5,10],
             'min_samples_leaf': [1,5,10],
             'bootstrap':[True, False],
             'criterion': ['entropy','gini']}

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(estimator= classifier, 
                          param_grid=parameters,
                          scoring= 'accuracy',
                         cv =10)

t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds"%(t1 - t0))



print(f'Best Accuracy : {rf_best_accuracy}')
print(f'Best Parameters : {rf_best_parameters}')
y_pred = grid_search.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))




df_cm = pd.DataFrame(cm, index=(0,1), columns=(0,1))
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')

