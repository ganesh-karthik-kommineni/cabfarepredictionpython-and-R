#!/usr/bin/env python
# coding: utf-8

# # Cab Fare Detection

# ### Problem Statement-
# You are a cab rental start-up company. You have successfully run the pilot project and now want to launch your cab service across the country. You have collected thehistorical data from your pilot project and now have a requirement to apply analytics forfare prediction. You need to design a system that predicts the fare amount for a cab ride in the city.

# In[1179]:


#loading libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')
from geopy.distance import geodesic
from geopy.distance import great_circle
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.externals import joblib 


# In[1180]:


#working directory
os.getcwd()


# The details of data attributes in the dataset are as follows:
# -  pickup_datetime - timestamp value indicating when the cab ride started.
# -  pickup_longitude - float for longitude coordinate of where the cab ride started.
# -  pickup_latitude - float for latitude coordinate of where the cab ride started.
# -  dropoff_longitude - float for longitude coordinate of where the cab ride ended.
# -  dropoff_latitude - float for latitude coordinate of where the cab ride ended.
# -  passenger_count - an integer indicating the number of passengers in the cab ride.

# predictive modeling machine learning project can be broken down into below workflow: 
# 1. Prepare Problem 
# a) Load libraries b) Load dataset 
# 2. Summarize Data a) Descriptive statistics b) Data visualizations 
# 3. Prepare Data a) Data Cleaning b) Feature Selection c) Data Transforms 
# 4. Evaluate Algorithms a) Split-out validation dataset b) Test options and evaluation metrics c) Spot Check Algorithms d) Compare Algorithms 
# 5. Improve Accuracy a) Algorithm Tuning b) Ensembles 
# 6. Finalize Model a) Predictions on validation dataset b) Create standalone model on entire training dataset c) Save model for future use

# In[1181]:


#importing data
train_data = pd.read_csv('train_cab.csv',dtype={'fare_amount':np.float64},na_values={'fare_amount':'430-'})
test_data = pd.read_csv('test.csv')
data = [train_data,test_data]
for i in data:
    i['pickup_datetime'] = pd.to_datetime(i['pickup_datetime'], errors = 'coerce')
train_data.head()


# In[1182]:


train_data.info()


# In[1183]:


test_data.head(5)


# In[1184]:


test_data.info()


# In[1185]:


test_data.describe()


# In[1186]:


train_data.describe()


# # Exploratory Data Analysis

# -  we will convert passenger_count into a categorical variable because passenger_count is not a continuous variable.
# -  passenger_count cannot take continous values, and also they are limited in number if its a cab.

# In[1187]:


cate_var=['passenger_count']
nume_var=['fare_amount','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']


# # Graphical Exploratory Data Analysis - Data Visualization

# In[1188]:


#making sns for plots
sns.set(style = 'whitegrid', palette = 'Set2' )


# In[1189]:


plt.figure(figsize=(20,20))
plt.subplot(321)
_ = sns.distplot(train_data['fare_amount'], bins=50)
plt.subplot(322)
_ = sns.distplot(train_data['pickup_longitude'], bins=50)
plt.subplot(323)
_ = sns.distplot(train_data['pickup_latitude'], bins=50)
plt.subplot(324)
_ = sns.distplot(train_data['dropoff_longitude'], bins=50)
plt.subplot(325)
_ = sns.distplot(train_data['dropoff_latitude'], bins=50)
plt.savefig('histogram.png')
plt.show()


# In[1190]:


# it's time waste as swarmplots are very slow
#plt.figure(figsize=(20,20))
#_ = sns.swarmplot(x='passenger_count',y='fare_amount',data=train_data)
#plt.title('Cab Fare w.r.t passenger_count')


# -  Jointplots for Bivariate Analysis.
# -  Here Scatter plot has regression line between 2 variables along with separate Bar plots of both variables.
# -  Also its annotated with pearson correlation coefficient and p value.

# In[1191]:


_ = sns.jointplot(x = 'fare_amount', y='pickup_longitude', data = train_data, kind = 'reg')
_.annotate(stats.pearsonr)
plt.savefig('jointfplo.png')
plt.show()


# In[1192]:


_ = sns.jointplot(x = 'fare_amount', y='pickup_latitude', data=train_data, kind='reg')
_.annotate(stats.pearsonr)
plt.savefig('jointfpla.png')
plt.show()


# In[1193]:


_ = sns.jointplot(x='fare_amount', y='dropoff_longitude', data=train_data, kind='reg')
_.annotate(stats.pearsonr)
plt.savefig('jointfdlo.png')
plt.show()


# In[1194]:


_ = sns.jointplot(x='fare_amount', y='dropoff_latitude', data=train_data, kind='reg')
_.annotate(stats.pearsonr)
plt.savefig('jointfdla.png')
plt.show()


# #### violin plots to see variable split

# In[1195]:


plt.figure(figsize=(20,20))
plt.subplot(321)
_ = sns.violinplot(y='fare_amount', data=train_data)
plt.subplot(322)
_ = sns.violinplot(y='pickup_longitude', data=train_data)
plt.subplot(323)
_ =sns.violinplot(y='pickup_latitude', data=train_data)
plt.subplot(324)
_ = sns.violinplot(y='dropoff_longitude', data=train_data)
plt.subplot(325)
_ = sns.violinplot(y='dropoff_latitude', data=train_data)
#plt.savefig('violinplot.png')
plt.show()


# #### now pair plots for numerical variables

# In[1196]:


_ =sns.pairplot(data=train_data[nume_var],kind='scatter',dropna=True)
_.fig.suptitle('Pairwise plot of numerical variables')
#plt.savefig('Pairwiseplot.png')
plt.show()


# ## Removing Outliers by basic understanding of dataset

# - Fare amount has a negative value, which doesn't make sense. A price amount cannot be -ve and also cannot be 0. So let's remove these fields.

# In[1197]:


sum(train_data['fare_amount']<1)


# In[1198]:


train_data[train_data['fare_amount']<1]


# In[1199]:


train_data = train_data.drop(train_data[train_data['fare_amount']<1].index, axis=0)


# In[1200]:


#train_data.loc[train_data['fare_amount'] < 1,'fare_amount'] = np.nan


#  - passenger_count variable

# In[1201]:


for i in range(4,11):
    print('passenger_count above' +str(i)+'={}'.format(sum(train_data['passenger_count']>i)))


# so 20 observations of passenger_count is consistenly above from 6,7,8,9,10 passenger_counts, let's see.

# In[1202]:


train_data[train_data['passenger_count']>6]


# lets see if there is passenger_count<1

# In[1203]:


train_data[train_data['passenger_count']<1]


# In[1204]:


len(train_data[train_data['passenger_count']<1])


# In[1205]:


test_data['passenger_count'].unique()


#  - Passenger_count variable contains values which are equal to 0.
# -  Test data does not contain passenger_count=0 . So if we do feature engineer to passenger_count of train dataset then it will create a dummy variable for passenger_count=0, which will be an additional feature compared to test dataset.
# -  So, let's remove those 0 values.
# -  And also let's remove 20 observation which are above 6 value because a cab cannot hold these number of passengers.

# In[1206]:


train_data = train_data.drop(train_data[train_data['passenger_count']>6].index, axis=0)
train_data = train_data.drop(train_data[train_data['passenger_count']<1].index, axis=0)


# In[1207]:


#train_data.loc[train_data['passenger_count'] >6,'passenger_count'] = np.nan
#train_data.loc[train_data['passenger_count'] >1,'passenger_count'] = np.nan


# In[1208]:


sum(train_data['passenger_count']>6)


#  - Latitudes range from -90 to 90.
#  - Longitudes range from -180 to 180. 
#  - Remove the ones which does not satisfy these ranges.

# In[1209]:


print('pickup_longitude above 180={}'.format(sum(train_data['pickup_longitude']>180)))
print('pickup_longitude below -180={}'.format(sum(train_data['pickup_longitude']<-180)))
print('pickup_latitude above 90={}'.format(sum(train_data['pickup_latitude']>90)))
print('pickup_latitude below -90={}'.format(sum(train_data['pickup_latitude']<-90)))
print('dropoff_longitude above 180={}'.format(sum(train_data['dropoff_longitude']>180)))
print('dropoff_longitude below -180={}'.format(sum(train_data['dropoff_longitude']<-180)))
print('dropoff_latitude below -90={}'.format(sum(train_data['dropoff_latitude']<-90)))
print('dropoff_latitude above 90={}'.format(sum(train_data['dropoff_latitude']>90)))


# -  Only one outlier found which is in the variable pickup_latitude.So let's remove it with nan.
# -  Also let's see if there are any values equal to 0.

# In[1210]:


for i in ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']:
    print(i,'equal to 0={}'.format(sum(train_data[i]==0)))


# let's remove the values which are equal to 0

# In[1211]:


train_data = train_data.drop(train_data[train_data['pickup_latitude']>90].index, axis=0)
for i in ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']:
    train_data = train_data.drop(train_data[train_data[i]==0].index, axis=0)


# In[1212]:


#for i in ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']:
 #   train_data.loc[train_data[i]==0,i] = np.nan
#train_data.loc[train_data['pickup_latitude']>90,'pickup_latitude'] = np.nan


# In[1213]:


train_data.shape


# So, we lost 16067-15661=406 observations

# In[1214]:


#df=train_data.copy()
train_data=df.copy()


# # Missing Value Analysis

# In[1215]:


#Create dataframe with missing percentage
missing_val = pd.DataFrame(train_data.isnull().sum())
#Reset index
missing_val = missing_val.reset_index()
missing_val


# -  As we can see there are some missing values in this data.
# -  pickup_datetime variable has 1 missing value. 
# -  We will impute missing values for fare_amount,passenger_count variables except pickup_datetime.
# -  And let's drop that 1 row which has missing value in pickup_datetime.

# In[1216]:


#Rename variable
missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})
missing_val
#Calculate percentage
missing_val['Missing_percentage'] = (missing_val['Missing_percentage']/len(train_data))*100
#descending order
missing_val = missing_val.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)
missing_val


# For Passenger_count:
# -  Actual value = 1
# -  Mode = 1
# -  KNN = 2

# In[1217]:


# Choose random values to replace it as NA
train_data['passenger_count'].loc[1000]


# In[1218]:


# Replacing 1.0 with NA
train_data['passenger_count'].loc[1000] = np.nan
train_data['passenger_count'].loc[1000]


# In[1219]:


# Impute with mode
train_data['passenger_count'].fillna(train_data['passenger_count'].mode()[0]).loc[1000]


# We can't use mode because data will be more biased towards passenger_count=1

# For fare_amount: 
# -  Actual value = 7.0
# -  Mean = 15.118
# -  Median = 8.5
# -  KNN = 7.369801

# In[1220]:


#for i in ['fare_amount','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']:
     # Choosing a random values to replace it as NA
 #    a=train_data[i].loc[1000]
  #   print(i,'at loc-1000:{}'.format(a))
     # Replacing 1.0 with NA
   #  train_data[i].loc[1000] = np.nan
    # print('Value after replacing with nan:{}'.format(train_data[i].loc[1000]))
     # Impute with mean
     #print('Value if imputed with mean:{}'.format(train_data[i].fillna(train_data[i].mean()).loc[1000]))
     # Impute with median
     #print('Value if imputed with median:{}\n'.format(train_data[i].fillna(train_data[i].median()).loc[1000]))


# In[1221]:


# Choosing a random values to replace it as NA
a=train_data['fare_amount'].loc[1000]
print('fare_amount at loc-1000:{}'.format(a))
# Replacing 1.0 with NA
train_data['fare_amount'].loc[1000] = np.nan
print('Value after replacing with nan:{}'.format(train_data['fare_amount'].loc[1000]))
# Impute with mean
print('Value if imputed with mean:{}'.format(train_data['fare_amount'].fillna(train_data['fare_amount'].mean()).loc[1000]))
# Impute with median
print('Value if imputed with median:{}'.format(train_data['fare_amount'].fillna(train_data['fare_amount'].median()).loc[1000]))


# In[1222]:


train_data.std()


# In[1223]:


columns=['fare_amount', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'passenger_count']


# we will separate pickup_datetime into a different dataframe and then merge with train_data in feature engineering step.

# In[1224]:


pickup_datetime=pd.DataFrame(train_data['pickup_datetime'])


# In[1225]:


# Imputing with missing values using KNN
train_data = pd.DataFrame(KNNImputer(n_neighbors = 7).fit_transform(train_data.drop('pickup_datetime',axis=1)),columns=columns, index=train_data.index)


# In[1226]:


train_data.std()


# In[1227]:


train_data.loc[1000]


# In[1228]:


train_data['passenger_count'].head()


# In[1229]:


train_data['passenger_count']=train_data['passenger_count'].astype('int')


# In[1230]:


train_data.std()


# In[1231]:


train_data['passenger_count'].unique()


# In[1232]:


train_data['passenger_count']=train_data['passenger_count'].round().astype('object').astype('category')


# In[1233]:


train_data['passenger_count'].unique()


# In[1234]:


train_data.loc[1000]


# - Now about missing value in pickup_datetime

# In[1235]:


pickup_datetime.head()


# In[1236]:


#Create dataframe with missing percentage
missing_val = pd.DataFrame(pickup_datetime.isnull().sum())
#Reset index
missing_val = missing_val.reset_index()
missing_val


# In[1237]:


pickup_datetime.shape


# In[1238]:


train_data.shape


# -  We will drop 1 row which has missing value for pickup_datetime variable
# after feature engineering step because if we drop now, pickup_datetime dataframe will have 16040 rows and our train_data has 1641 rows, then if we merge these two dataframes then pickup_datetime variable will gain one missing value.
# - If we merge and then drop now then we would require to split again before outlier analysis and then merge again in feature engineering step.
# -  So, instead of doing the work two times we will drop it one time i.e. after feature engineering process.

# In[1239]:


df1 = train_data.copy()
#train_data = df1.copy()


# In[1240]:


train_data['passenger_count'].describe()


# In[1241]:


train_data.describe()


# ## Outlier Analysis using Boxplot
# -  We Will do Outlier Analysis only on Fare_amount just for now and we will do outlier analysis after feature engineering laitudes and longitudes.

# -  Univariate Boxplots: Boxplots for all Numerical Variables including target variable.

# In[1242]:


plt.figure(figsize=(20,5)) 
plt.xlim(0,100)
sns.boxplot(x=train_data['fare_amount'],data=train_data,orient='h')
plt.title('Boxplot of fare_amount')
#plt.savefig('boxplot of fare_amount.png')
plt.show()


# In[1243]:


#sum(train_data['fare_amount']<22.5)/len(train_data['fare_amount'])*100


# -  Bivariate Boxplots: Boxplot for Numerical Variable Vs Categorical Variable.

# In[1244]:


plt.figure(figsize=(20,10))
plt.xlim(0,100)
_ = sns.boxplot(x=train_data['fare_amount'],y=train_data['passenger_count'],data=train_data,orient='h')
plt.title('Boxplot of fare_amount w.r.t passenger_count')
plt.savefig('Boxplot of fare_amount w.r.t passenger_count.png')
plt.show()


# In[1245]:


train_data.describe()


# In[1246]:


train_data['passenger_count'].describe()


# ## Outlier Treatment
# -  From the above Boxplots we can say that there are outliers in the train dataset.
# -  Reconsider pickup_longitude,etc.

# In[1247]:


def outliers_treatment(col):
    
    ##calculating outlier indices and replace them with NA  
    #Extract quartiles
    
    q75, q25 = np.percentile(train_data[col], [75 ,25])
    print(q75,q25)
    
    #Calculate IQR
    
    iqr = q75 - q25
    
    #Calculate inner and outer fence
    
    minimum = q25 - (iqr*1.5)
    maximum = q75 + (iqr*1.5)
    print(minimum,maximum)
    
    #Replacing with NA
    
    train_data.loc[train_data[col] < minimum,col] = np.nan
    train_data.loc[train_data[col] > maximum,col] = np.nan


# In[1248]:


for i in nume_var:
    outliers_treatment('fare_amount')
    #outliers_treatment('pickup_longitude')
    #outliers_treatment('pickup_latitude')
    #outliers_treatment('dropoff_longitude')
    #outliers_treatment('dropoff_latitude')


# In[1249]:


pd.DataFrame(train_data.isnull().sum())


# In[1250]:


train_data.std()


# In[1251]:


#Imputing with missing values using KNN
train_data = pd.DataFrame(KNNImputer(n_neighbors = 7).fit_transform(train_data), columns = train_data.columns, index=train_data.index)


# In[1252]:


train_data.std()


# In[1253]:


train_data['passenger_count'].describe()


# In[1254]:


train_data['passenger_count']=train_data['passenger_count'].astype('int').round().astype('object').astype('category')


# In[1255]:


train_data.describe()


# In[1256]:


train_data.head()


# In[1257]:


df2 = train_data.copy()
#train=df2.copy()


# In[1258]:


train_data.shape


# ## Feature Engineering

# #### Feature Engineering for timestamp variable
# -  Let's us derive new features from pickup_datetime variable
# -  New features will be year,month,day_of_week,hour

# In[1259]:


# Let's Join 2 Dataframes pickup_datetime and train
train_data = pd.merge(pickup_datetime,train_data,right_index=True,left_index=True)
train_data.head()


# In[1260]:


train_data.shape


# In[1261]:


train_data = train_data.reset_index(drop=True)


# - As we discussed in Missing value imputation step about dropping the missing value, let's do it now.

# In[1262]:


pd.DataFrame(train_data.isna().sum())


# In[1263]:


train_data = train_data.dropna()


# In[1264]:


data = [train_data,test_data]
for i in data:
    i["year"] = i["pickup_datetime"].apply(lambda row: row.year)
    i["month"] = i["pickup_datetime"].apply(lambda row: row.month)
    #i["day_of_month"] = i["pickup_datetime"].apply(lambda row: row.day)
    i["day_of_week"] = i["pickup_datetime"].apply(lambda row: row.dayofweek)
    i["hour"] = i["pickup_datetime"].apply(lambda row: row.hour)


# In[1265]:


train_nodummies=train_data.copy()
# train_data=train_nodummies.copy()


# In[1266]:


plt.figure(figsize=(20,10))
sns.countplot(train_data['year'])
plt.savefig('year.png')

plt.figure(figsize=(20,10))
sns.countplot(train_data['month'])
plt.savefig('month.png')

plt.figure(figsize=(20,10))
sns.countplot(train_data['day_of_week'])
plt.savefig('day_of_week.png')

plt.figure(figsize=(20,10))
sns.countplot(train_data['hour'])
plt.savefig('hour.png')


# Now let's use month,day_of_week,hour to derive new features like sessions in a day, seasons in a year, week:weekend/weekday

# In[1267]:


def f(x):
    ###for sessions in a day using hour column
    if (x >=5) and (x <= 11):
        return 'morning'
    elif (x >=12) and (x <=16 ):
        return 'afternoon'
    elif (x >= 17) and (x <= 20):
        return'evening'
    elif (x >=21) and (x <= 23) :
        return 'night_PM'
    elif (x >=0) and (x <=4):
        return'night_AM'


# In[1268]:


def g(x):
    ###for seasons in a year using month column
    if (x >=3) and (x <= 5):
        return 'spring'
    elif (x >=6) and (x <=8 ):
        return 'summer'
    elif (x >= 9) and (x <= 11):
        return'fall'
    elif (x >=12)|(x <= 2) :
        return 'winter'


# In[1269]:


def h(x):
    ###for week:weekday/weekend in a day_of_week column
    if (x >=0) and (x <= 4):
        return 'weekday'
    elif (x >=5) and (x <=6 ):
        return 'weekend'


# In[1270]:


train_data['session'] = train_data['hour'].apply(f)
test_data['session'] = test_data['hour'].apply(f)
#train_nodummies['session'] = train_nodummies['hour'].apply(f)


# In[1271]:


train_data['seasons'] = train_data['month'].apply(g)
test_data['seasons'] = test_data['month'].apply(g)
#train_data['seasons'] = test_data['month'].apply(g)


# In[1272]:


train_data['week'] = train_data['day_of_week'].apply(h)
test_data['week'] = test_data['day_of_week'].apply(h)


# In[1273]:


train_data.shape


# In[1274]:


test_data.shape


# #### Feature Engineering for passenger_count variable
# - Because models in scikit learn require numerical input. if dataset contains categorical variables then we have to encode them.
# - Let's use one hot encoding technique for passenger_count variable.

# In[1275]:


train_data['passenger_count'].describe()


# In[1276]:


#Creating dummies for each variable in passenger_count and merging dummies dataframe to both train and test dataframe
temp = pd.get_dummies(train_data['passenger_count'], prefix = 'passenger_count')
train_data = train_data.join(temp)
temp = pd.get_dummies(test_data['passenger_count'], prefix = 'passenger_count')
test_data = test_data.join(temp)
temp = pd.get_dummies(train_data['seasons'], prefix = 'season')
train_data = train_data.join(temp)
temp = pd.get_dummies(test_data['seasons'], prefix = 'season')
test_data = test_data.join(temp)
temp = pd.get_dummies(train_data['week'], prefix = 'week')
train_data = train_data.join(temp)
temp = pd.get_dummies(test_data['week'], prefix = 'week')
test_data = test_data.join(temp)
temp = pd.get_dummies(train_data['session'], prefix = 'session')
train_data = train_data.join(temp)
temp = pd.get_dummies(test_data['session'], prefix = 'session')
test_data = test_data.join(temp)
temp = pd.get_dummies(train_data['year'], prefix = 'year')
train_data = train_data.join(temp)
temp = pd.get_dummies(test_data['year'], prefix = 'year')
test_data = test_data.join(temp)


# In[1277]:


train_data.head()


# In[1278]:


test_data.head()


# Let's drop one column from each one-hot-encoded variables

# In[1279]:


train_data.columns


# In[1280]:


train_data = train_data.drop(['passenger_count_1','season_fall','week_weekday','session_afternoon','year_2009'],axis=1)
test_data=test_data.drop(['passenger_count_1','season_fall','week_weekday','session_afternoon','year_2009'],axis=1)


# #### Feature Engineering for latitude and longitude variable
# -  As we have latitude and longitude data for pickup and dropoff, let's find the distance the cab travelled from pickup and dropoff location.

# In[1281]:


#train_data.sort_values('pickup_datetime')


# In[1282]:


#def haversine(coord1, coord2):
     ###Calculate distance the cab travelled from pickup and dropoff location using the Haversine Formula
 #    data = [train_data, test_data]
  #   for i in data:
   #      lon1, lat1 = coord1
    #     lon2, lat2 = coord2
     #    R = 6371000  # radius of Earth in meters
      #   phi_1 = np.radians(i[lat1])
       #  phi_2 = np.radians(i[lat2])
        # delta_phi = np.radians(i[lat2] - i[lat1])
         #delta_lambda = np.radians(i[lon2] - i[lon1])
         #a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi_1) * np.cos(phi_2) * np.sin(delta_lambda / 2.0) ** 2
        # c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
         #meters = R * c  # output distance in meters
       #  km = meters / 1000.0  # output distance in kilometers
        # miles = round(km, 3)/1.609344
        # i['distance'] = miles
    # print(f"Distance: {miles} miles")
    # return miles


# In[1283]:


#haversine(['pickup_longitude','pickup_latitude'],['dropoff_longitude','dropoff_latitude'])


# In[1284]:


# Calculate distance the cab travelled from pickup and dropoff location using great_circle from geopy library
data = [train_data, test_data]
for i in data:
    i['great_circle']=i.apply(lambda x: great_circle((x['pickup_latitude'],x['pickup_longitude']), (x['dropoff_latitude'],   x['dropoff_longitude'])).miles, axis=1)
    i['geodesic']=i.apply(lambda x: geodesic((x['pickup_latitude'],x['pickup_longitude']), (x['dropoff_latitude'],   x['dropoff_longitude'])).miles, axis=1)


# In[1285]:


train_data.head()


# In[1286]:


test_data.head()


# As Vincenty is more accurate than haversine. Also vincenty is prefered for short distances.
# - Therefore let's drop great_circle. 
# - let's drop them together with other variables which were used to feature engineer.

# In[1287]:


pd.DataFrame(train_data.isna().sum())


# In[1288]:


pd.DataFrame(test_data.isna().sum())


# #### Let's remove the variables which were used to feature engineer new variables

# In[1289]:


#train_nodummies=train_nodummies.drop(['pickup_datetime','pickup_longitude', 'pickup_latitude',
 #       'dropoff_longitude', 'dropoff_latitude','great_circle'], axis = 1, inplace = True)
#test_nodummies=test_data.drop(['pickup_datetime','pickup_longitude', 'pickup_latitude',
 #       'dropoff_longitude', 'dropoff_latitude','passenger_count_1', 'passenger_count_2', 'passenger_count_3',
 #       'passenger_count_4', 'passenger_count_5', 'passenger_count_6',
 #       'season_fall', 'season_spring', 'season_summer', 'season_winter',
 #       'week_weekday', 'week_weekend', 'session_afternoon', 'session_evening',
  #      'session_morning', 'session_night (AM)', 'session_night (PM)',
  #      'year_2009', 'year_2010', 'year_2011', 'year_2012', 'year_2013',
  #      'year_2014', 'year_2015', 'great_circle'], axis = 1, inplace = True)


# In[1290]:


train_data=train_data.drop(['pickup_datetime','pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'year',
       'month', 'day_of_week', 'hour', 'session', 'seasons', 'week','great_circle'],axis=1)
test_data=test_data.drop(['pickup_datetime','pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'year',
       'month', 'day_of_week', 'hour', 'session', 'seasons', 'week','great_circle'],axis=1)


# In[1291]:


train_data.shape


# In[1292]:


test_data.shape


# In[1293]:


train_data.columns


# In[1294]:


test_data.columns


# In[1295]:


train_data.head()


# In[1296]:


test_data.head()


# In[1297]:


plt.figure(figsize=(20,5)) 
sns.boxplot(x=train_data['geodesic'],data=train_data,orient='h')
plt.title('Boxplot of geodesic ')
#plt.savefig('boxplot of geodesic.png')
plt.show()


# In[1298]:


plt.figure(figsize=(20,5)) 
plt.xlim(0,100)
sns.boxplot(x=train_data['geodesic'],data=train_data,orient='h')
plt.title('Boxplot of geodesic ')
#plt.savefig('boxplot of geodesic.png')
plt.show()


# In[1299]:


outliers_treatment('geodesic')


# In[1300]:


pd.DataFrame(train_data.isnull().sum())


# In[1301]:


#Imputing with missing values using KNN
train_data = pd.DataFrame(KNNImputer(n_neighbors = 7).fit_transform(train_data), columns = train_data.columns, index=train_data.index)


# ## Feature Selection
# 1.Correlation Analysis
# 
#     Statistically correlated: features move together directionally.
#     Linear models assume feature independence, and if features are correlated that could introduce bias into our models.

# In[1302]:


cate_var=['passenger_count_2',
       'passenger_count_3', 'passenger_count_4', 'passenger_count_5',
       'passenger_count_6', 'season_spring', 'season_summer',
       'season_winter', 'week_weekend',
       'session_evening', 'session_morning', 'session_night_AM',
       'session_night_PM', 'year_2010', 'year_2011',
       'year_2012', 'year_2013', 'year_2014', 'year_2015']
nume_var=['fare_amount','geodesic']
train_data[cate_var]=train_data[cate_var].apply(lambda x: x.astype('category') )
test_data[cate_var]=test_data[cate_var].apply(lambda x: x.astype('category') ) 


# - Let's plot a Heatmap of correlation whereas correlation measures how strongly 2 quantities are related to each other.

# In[1303]:


# heatmap using correlation matrix
plt.figure(figsize=(15,15))
_ = sns.heatmap(train_data[nume_var].corr(), square=True, cmap='RdYlGn',linewidths=0.5,linecolor='w',annot=True)
plt.title('Correlation matrix ')
plt.savefig('correlation.png')
plt.show()


# As we can see from above correlation plot fare_amount and geodesic are correlated to each other.
# 
# - Jointplots for Bivariate Analysis.
# - Here Scatter plot has regression line between two variables along with separate Bar plots of both variables.
# - Also it's annotated with pearson correlation coefficient and p value.

# In[1304]:


_ = sns.jointplot(x='fare_amount',y='geodesic',data=train_data,kind = 'reg')
_.annotate(stats.pearsonr)
plt.savefig('jointct.png')
plt.show()


# ## Chi-square test of Independence for Categorical Variables/Features
# - Hypothesis testing :
#    - Null Hypothesis: 2 variables are independent.
#    - Alternate Hypothesis: 2 variables are not independent.
# - If p-value is less than 0.05 then we reject the null hypothesis saying that two variables are dependent.
# - And if p-value is greater than 0.05 then we accept the null hypothesis saying that two variables are independent.
# - There should be no dependencies between Independent variables.
# - So let's remove that variable whose p-value with other variable is low than 0.05.
# - And let's keep that variable whose p-value with other variable is high than 0.05

# In[1305]:


#loop for chi square values
for i in cate_var:
    for j in cate_var:
        if(i != j):
            chi2, p, dof, ex = chi2_contingency(pd.crosstab(train_data[i], train_data[j]))
            if(p < 0.05):
                print(i,"and",j,"are dependent on each other with",p,'----Remove')
            else:
                print(i,"and",j,"are independent on each other with",p,'----Keep')


# ## Analysis of Variance(Anova) Test
# -  It is carried out to compare between each groups in a categorical variable.
# -  ANOVA only lets us know the means for different groups are same or not. It doesn’t help us identify which mean is different.
# -  Hypothesis testing :
#     -  Null Hypothesis: mean of all categories in a variable are same.
#     -  Alternate Hypothesis: mean of at least one category in a variable is different.
# -  If p-value is less than 0.05 then we reject the null hypothesis.
# -  And if p-value is greater than 0.05 then we accept the null hypothesis.

# In[1306]:


train_data.columns


# In[1307]:


#ANOVA _1)+C(passenger_count_2)+C(passenger_count_3)+C(passenger_count_4)+C(passenger_count_5)+C(passenger_count_6)
model = ols('fare_amount ~ C(passenger_count_2)+C(passenger_count_3)+C(passenger_count_4)+C(passenger_count_5)+C(passenger_count_6)+C(season_spring)+C(season_summer)+C(season_winter)+C(week_weekend)+C(session_night_AM)+C(session_night_PM)+C(session_evening)+C(session_morning)+C(year_2010)+C(year_2011)+C(year_2012)+C(year_2013)+C(year_2014)+C(year_2015)',data=train_data).fit()
                
aov_table = sm.stats.anova_lm(model)
aov_table


# Every variable has p-value less than 0.05 therefore we reject the null hypothesis.

# ## Multicollinearity Test
# -  VIF is always greater or equal to 1.
# -  if VIF is 1 --- Not correlated to any of the variables.
# -  if VIF is between 1-5 --- Moderately correlated.
# -  if VIF is above 5 --- Highly correlated.
# -  If there are multiple variables with VIF greater than 5, only remove the variable with the highest VIF.

# In[1308]:


# _1+passenger_count_2+passenger_count_3+passenger_count_4+passenger_count_5+passenger_count_6
outcome, predictors = dmatrices('fare_amount ~ geodesic+passenger_count_2+passenger_count_3+passenger_count_4+passenger_count_5+passenger_count_6+season_spring+season_summer+season_winter+week_weekend+session_night_AM+session_night_PM+session_evening+session_morning+year_2010+year_2011+year_2012+year_2013+year_2014+year_2015',train_data, return_type='dataframe')
# calculating VIF for each individual Predictors
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(predictors.values, i) for i in range(predictors.shape[1])]
vif["features"] = predictors.columns
vif


# So we have no or very low multicollinearity

# ## Feature Scaling Check with or without normalization of standard scalar

# In[1309]:


train_data[nume_var].var()


# In[1310]:


sns.distplot(train_data['geodesic'],bins=50)
#plt.savefig('distplot.png')


# In[1311]:


plt.figure()
stats.probplot(train_data['geodesic'], dist='norm', fit=True,plot=plt)
#plt.savefig('qq prob plot.png')


# In[1312]:


#Normalization
train_data['geodesic'] = (train_data['geodesic'] - min(train_data['geodesic']))/(max(train_data['geodesic']) - min(train_data['geodesic']))
test_data['geodesic'] = (test_data['geodesic'] - min(test_data['geodesic']))/(max(test_data['geodesic']) - min(test_data['geodesic']))


# In[1313]:


train_data['geodesic'].var()


# In[1314]:


sns.distplot(train_data['geodesic'],bins=50)
#plt.savefig('distplot.png')


# In[1354]:


plt.figure()
stats.probplot(train_data['geodesic'], dist='norm', fit=True,plot=plt)
plt.savefig('qq prob plot1.png')


# In[1316]:


train_data.columns


# In[1319]:


#df4=train_data.copy()
train_data=df4.copy()
#f4=test_data.copy()
test_data=f4.copy()


# In[1318]:


train_data = train_data.drop(['passenger_count_2'],axis=1)
test_data = test_data.drop(['passenger_count_2'],axis=1)


# In[1320]:


train_data.columns


# ## Splitting train into train and validation subsets
# - X_train y_train--are train subset
# - X_test y_test--are validation subset

# In[1321]:


X = train_data.drop('fare_amount',axis=1).values
y = train_data['fare_amount'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)
print(train_data.shape, X_train.shape, X_test.shape,y_train.shape,y_test.shape)


# In[1322]:


def rmsle(y,y_):
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))
def scores(y, y_):
    print('r square  ', metrics.r2_score(y, y_))
    print('Adjusted r square:{}'.format(1 - (1-metrics.r2_score(y, y_))*(len(y)-1)/(len(y)-X_train.shape[1]-1)))
    print('MAPE:{}'.format(np.mean(np.abs((y - y_) / y))*100))
    print('MSE:', metrics.mean_squared_error(y, y_))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y, y_))) 
def test_scores(model):
    print('Training Data Score')
    print()
    
    #Predicting result on Training data
    
    y_pred = model.predict(X_train)
    scores(y_train,y_pred)
    print('RMSLE:',rmsle(y_train,y_pred))
    print()
    print('Test Data Score')
    print()
    
    # Evaluating on Test Set
    
    y_pred = model.predict(X_test)
    scores(y_test,y_pred)
    print('RMSLE:',rmsle(y_test,y_pred))


# # Multiple Linear Regression

# In[1323]:


# Setup the parameters and distributions to sample from: param_dist

param_dist = {'copy_X':[True, False],
          'fit_intercept':[True,False]}

# Instantiate a Decision reg classifier: reg

reg = LinearRegression()

# Instantiate the gridSearchCV object: reg_cv

reg_cv = GridSearchCV(reg, param_dist, cv=5,scoring='r2')

# Fit it to the data

reg_cv.fit(X, y)

# Print the tuned parameters and score

print("Tuned Decision reg Parameters: {}".format(reg_cv.best_params_))
print("Best score is {}".format(reg_cv.best_score_))


# In[1324]:


# Create the regressor: reg_all

reg_all = LinearRegression(copy_X= True, fit_intercept=True)

# Fit the regressor to the training data

reg_all.fit(X_train,y_train)

# Predict on the test data: y_pred

y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE

print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))
test_scores(reg_all)

# Compute and print the coefficients

reg_coef = reg_all.coef_
print(reg_coef)

# Plot the coefficients

plt.figure(figsize=(15,5))
plt.plot(range(len(test.columns)), reg_coef)
plt.xticks(range(len(test.columns)), test.columns.values, rotation=60)
plt.margins(0.02)
#plt.savefig('linear coefficients')
plt.show()


# In[1325]:


from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg

reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores

cv_scores = cross_val_score(reg,X,y,cv=5,scoring='neg_mean_squared_error')

# Print the 5-fold cross-validation scores

print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))


# # Ridge Regression

# In[1326]:


# Setup the parameters and distributions to sample from: param_dist

param_dist = {'alpha':np.logspace(-4, 0, 50),
          'normalize':[True,False],
             'max_iter':range(500,5000,500)}

# Instantiate a Decision ridge classifier: ridge

ridge = Ridge()

# Instantiate the gridSearchCV object: ridge_cv

ridge_cv = GridSearchCV(ridge, param_dist, cv=5,scoring='r2')

# Fit it to the data

ridge_cv.fit(X, y)

# Print the tuned parameters and score

print("Tuned Decision ridge Parameters: {}".format(ridge_cv.best_params_))
print("Best score is {}".format(ridge_cv.best_score_))


# In[1327]:


# Instantiate a ridge regressor: ridge

ridge = Ridge(alpha=0.0005428675439323859, normalize=True,max_iter = 500)

# Fit the regressor to the data

ridge.fit(X_train,y_train)

# Compute and print the coefficients

ridge_coef = ridge.coef_

print(ridge_coef)

# Plot the coefficients

plt.figure(figsize=(15,5))

plt.plot(range(len(test.columns)), ridge_coef)

plt.xticks(range(len(test.columns)), test.columns.values, rotation=60)
plt.margins(0.02)
#plt.savefig('ridge coefficients')
plt.show()
test_scores(ridge)


# # Lasso Regression

# In[1328]:


# Setup the parameters and distributions to sample from: param_dist
param_dist = {'alpha':np.logspace(-4, 0, 50),
          'normalize':[True,False],
             'max_iter':range(500,5000,500)}
# Instantiate a Decision lasso classifier: lasso
lasso = Lasso()

# Instantiate the gridSearchCV object: lasso_cv
lasso_cv = GridSearchCV(lasso, param_dist, cv=5,scoring='r2')

# Fit it to the data
lasso_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Decision lasso Parameters: {}".format(lasso_cv.best_params_))
print("Best score is {}".format(lasso_cv.best_score_))


# In[1329]:


# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.00021209508879201905, normalize=False,max_iter = 500)

# Fit the regressor to the data
lasso.fit(X,y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)

# Plot the coefficients
plt.figure(figsize=(15,5))
plt.ylim(-1,10)
plt.plot(range(len(test.columns)), lasso_coef)
plt.xticks(range(len(test.columns)), test.columns.values, rotation=60)
plt.margins(0.02)
plt.savefig('lasso coefficients')
plt.show()
test_scores(lasso)


# # Decision Tree Regression

# In[1330]:


# Setup the parameters and distributions to sample from: param_dist
param_dist = {'max_depth': range(2,16,2),
              'min_samples_split': range(2,16,2)}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeRegressor()

# Instantiate the gridSearchCV object: tree_cv
tree_cv = GridSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))


# In[1331]:


# Instantiate a tree regressor: tree
tree = DecisionTreeRegressor(max_depth= 6, min_samples_split=2)

# Fit the regressor to the data
tree.fit(X_train,y_train)

# Compute and print the coefficients
tree_features = tree.feature_importances_
print(tree_features)

# Sort test importances in descending order
indices = np.argsort(tree_features)[::1]

# Rearrange test names so they match the sorted test importances
names = [test.columns[i] for i in indices]

# Creating plot
fig = plt.figure(figsize=(20,10))
plt.title("test Importance")

# Add horizontal bars
plt.barh(range(pd.DataFrame(X_train).shape[1]),tree_features[indices],align = 'center')
plt.yticks(range(pd.DataFrame(X_train).shape[1]), names)
plt.savefig('tree test importance')
plt.show()
# Make predictions and cal error
test_scores(tree)


# # Random Forest Regression

# In[1332]:


# Create the random grid
random_grid = {'n_estimators': range(100,500,100),
               'max_depth': range(5,20,1),
               'min_samples_leaf':range(2,5,1),
              'max_features':['auto','sqrt','log2'],
              'bootstrap': [True, False],
              'min_samples_split': range(2,5,1)}
# Instantiate a Decision Forest classifier: Forest
Forest = RandomForestRegressor()

# Instantiate the gridSearchCV object: Forest_cv
Forest_cv = RandomizedSearchCV(Forest, random_grid, cv=5)

# Fit it to the data
Forest_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Random Forest Parameters: {}".format(Forest_cv.best_params_))
print("Best score is {}".format(Forest_cv.best_score_))


# In[1333]:


# Instantiate a Forest regressor: Forest
Forest = RandomForestRegressor(n_estimators=100, min_samples_split= 2, min_samples_leaf=4, max_features='auto', max_depth=9, bootstrap=True)

# Fit the regressor to the data
Forest.fit(X_train,y_train)

# Compute and print the coefficients
Forest_features = Forest.feature_importances_
print(Forest_features)

# Sort feature importances in descending order
indices = np.argsort(Forest_features)[::1]

# Rearrange feature names so they match the sorted feature importances
names = [test.columns[i] for i in indices]

# Creating plot
fig = plt.figure(figsize=(20,10))
plt.title("Feature Importance")

# Add horizontal bars
plt.barh(range(pd.DataFrame(X_train).shape[1]),Forest_features[indices],align = 'center')
plt.yticks(range(pd.DataFrame(X_train).shape[1]), names)
plt.savefig('Random forest feature importance')
plt.show()# Make predictions
test_scores(Forest)


# In[1334]:


from sklearn.model_selection import cross_val_score
# Create a random forest regression object: Forest
Forest = RandomForestRegressor(n_estimators=400, min_samples_split= 2, min_samples_leaf=4, max_features='auto', max_depth=12, bootstrap=True)

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(Forest,X,y,cv=5,scoring='neg_mean_squared_error')

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))


# # Improving Accuracy using XGBOOST

# #### Improve Accuracy
# - Algorithm Tuning
# - Ensembles

# In[1335]:


data_dmatrix = xgb.DMatrix(data=X,label=y)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)


# In[1336]:


dtrain,dtest,data_dmatrix


# In[1337]:


params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=5,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
cv_results.head()


# In[1338]:


# the final boosting round metric
print((cv_results["test-rmse-mean"]).tail(1))


# In[1340]:


Xgb = XGBRegressor()
Xgb.fit(X_train,y_train)
#pred_xgb = model_xgb.predict(X_test)
test_scores(Xgb)


# In[1341]:


# Create the random grid
para = {'n_estimators': range(100,500,100),
               'max_depth': range(3,10,1),
        'reg_alpha':np.logspace(-4, 0, 50),
        'subsample': np.arange(0.1,1,0.2),
        'colsample_bytree': np.arange(0.1,1,0.2),
        'colsample_bylevel': np.arange(0.1,1,0.2),
        'colsample_bynode': np.arange(0.1,1,0.2),
       'learning_rate': np.arange(.05, 1, .05)}
# Instantiate a Decision Forest classifier: Forest
Xgb = XGBRegressor()

# Instantiate the gridSearchCV object: Forest_cv
xgb_cv = RandomizedSearchCV(Xgb, para, cv=5)

# Fit it to the data
xgb_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Xgboost Parameters: {}".format(xgb_cv.best_params_))
print("Best score is {}".format(xgb_cv.best_score_))


# In[1342]:


# Instantiate a xgb regressor: xgb
Xgb = XGBRegressor(subsample= 0.1, reg_alpha= 0.08685113737513521, n_estimators= 200, max_depth= 3, learning_rate=0.05, colsample_bytree= 0.7000000000000001, colsample_bynode=0.7000000000000001, colsample_bylevel=0.9000000000000001)

# Fit the regressor to the data
Xgb.fit(X_train,y_train)

# Compute and print the coefficients
xgb_features = Xgb.feature_importances_
print(xgb_features)

# Sort feature importances in descending order
indices = np.argsort(xgb_features)[::1]

# Rearrange feature names so they match the sorted feature importances
names = [test.columns[i] for i in indices]

# Creating plot
fig = plt.figure(figsize=(20,10))
plt.title("Feature Importance")

# Add horizontal bars
plt.barh(range(pd.DataFrame(X_train).shape[1]),xgb_features[indices],align = 'center')
plt.yticks(range(pd.DataFrame(X_train).shape[1]), names)
plt.savefig(' xgb feature importance')
plt.show()# Make predictions
test_scores(Xgb)


# ## Finalizing the model
# - Create standalone model on entire training dataset
# - Save the model for later use

# In[1343]:


def rmsle(y,y_):
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))
def score(y, y_):
    print('r square  ', metrics.r2_score(y, y_))
    print('Adjusted r square:{}'.format(1 - (1-metrics.r2_score(y, y_))*(len(y)-1)/(len(y)-X_train.shape[1]-1)))
    print('MAPE:{}'.format(np.mean(np.abs((y - y_) / y))*100))
    print('MSE:', metrics.mean_squared_error(y, y_))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y, y_)))
    print('RMSLE:',rmsle(y_test,y_pred))
def scores(model):
    print('Training Data Score')
    print()
    #Predicting result on Training data
    y_pred = model.predict(X)
    score(y,y_pred)
    print('RMSLE:',rmsle(y,y_pred))   


# In[1344]:


test_data.columns


# In[1345]:


train_data.columns


# In[1346]:


train_data.shape


# In[1348]:


test_data.shape


# In[1349]:


a = pd.read_csv('test.csv')


# In[1350]:


test_pickup_datetime = a['pickup_datetime']


# In[1351]:


# Instantiate a xgb regressor: xgb
Xgb = XGBRegressor(subsample= 0.1, reg_alpha= 0.08685113737513521, n_estimators= 200, max_depth= 3, learning_rate=0.05, colsample_bytree= 0.7000000000000001, colsample_bynode=0.7000000000000001, colsample_bylevel=0.9000000000000001)

# Fit the regressor to the data
Xgb.fit(X,y)

# Compute and print the coefficients
xgb_features = Xgb.feature_importances_
print(xgb_features)

# Sort feature importances in descending order
indices = np.argsort(xgb_features)[::1]

# Rearrange feature names so they match the sorted feature importances
names = [test.columns[i] for i in indices]

# Creating plot
fig = plt.figure(figsize=(20,10))
plt.title("Feature Importance")

# Add horizontal bars
plt.barh(range(pd.DataFrame(X_train).shape[1]),xgb_features[indices],align = 'center')
plt.yticks(range(pd.DataFrame(X_train).shape[1]), names)
plt.savefig(' xgb1 feature importance')
plt.show()
scores(Xgb)

# Predictions
pred = Xgb.predict(test.values)
pred_results_wrt_date = pd.DataFrame({"pickup_datetime":test_pickup_datetime,"fare_amount" : pred})
pred_results_wrt_date.to_csv("predictions_xgboost.csv",index=False)


# In[1352]:


pred_results_wrt_date


# In[1353]:


# Save the model as a pickle in a file 
joblib.dump(Xgb, 'cab_fare_xgboost_model.pkl') 
  
# # Load the model from the file 
# Xgb_from_joblib = joblib.load('cab_fare_xgboost_model.pkl')  


# In[ ]:




