
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/content/Titanic-Dataset.csv')

"""exploring the dataset"""

df.head()

df.info()

df.describe()

df.isnull().sum()

df.nunique()

df.duplicated().sum()

df['Embarked'].value_counts()

"""handling the missing data"""

#filling the age colum with the median value why median bacause the data may contain outlier values
df['Age'].fillna(df['Age'].median(),inplace=True)

#filling embarked with mode (mode can be used for categotical columns )
#(why[0]- mode return a frequent values to get the first frequent value we use this [0] )

df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)

#drop cabin (too many null values)

df.drop('Cabin',axis = 1,inplace=True)

# or we can fill this code with the values like unknown

df['cabin'].fillna("unknown")

"""Encoding the categorical variables"""

#label encoding sex column assigning o for male and 1 for female
df['Sex'] = df['Sex'].map({'male':0,'female':1})

#using one hot encoding because it is a location

df = pd.get_dummies(df,columns=['Embarked'],drop_first=True)

"""normalizing"""

#normalizing the age and fare column between 0 to 1 to improve the model performance
#using standardscaler we can use other preprocessors like minmaxscaler but it is more suitable in timeseries data,robustscaler when handling data with outlier values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['Age', 'Fare']] =scaler.fit_transform(df[['Age', 'Fare']])

"""detect,remove outlier values"""

sns.boxplot(x=df['Fare'])
plt.show()

# there are several methods to remove outlier some of the methods like IQR z-score and isolation forest can be used for this data
#i am using the z-score method where - a value is more than 3 standard deviations from the mean, it’s an outlier.

from scipy import stats
z = np.abs(stats.zscore(df['Fare']))
df = df[(z<3)]

df.head()

"""covert the cleaned data to a csv file"""

#convert the data into csv usin the to_csv function in pandas

df.to_csv("titanic_cleaned_data_by_sivashankar.csv", index=False)