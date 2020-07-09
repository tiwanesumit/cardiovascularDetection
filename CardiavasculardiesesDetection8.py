#To build an application to classify the patients to be healthy or suffering from
 #cardiovascular disease based on the given attributes.
#import numpy as np
"""  
     HACKATHON
     
"""     
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_excel('D:/Data/cardio_train.xlsx')
df.head()
df.info()
df.describe().T
df.isnull().sum()
#coverting date into year
df['age'] = df['age']/360
#df['age'] = pd.to_numeric(df['age'], downcast='float')
df['age'] = df['age'].astype(int)
df['age'].head()
df['height'] = df['height']/100
df['height1'] = df['height'] * df['height']
df['BMI'] = df['weight'] / df['height1'] 
#weight/meter^2 kg/m^2

df.head()

df.BMI.value_counts()
df.age.value_counts()
df.height.value_counts()
df.weight.value_counts()
df.ap_lo.value_counts()
df.ap_hi.value_counts()

#visual for outliers

sns.boxplot(x=df['ap_lo'])

"""
Z-Score-
Wikipedia Definition
The Z-score is the signed number of standard deviations by which the value
of an observation or data point is above the mean value of 
what is being observed or measured.


"""

from scipy import stats
z = np.abs(stats.zscore(df))
print(z)


"""
Looking the code and the output above, it is difficult to say
 which data point is an outlier. Let’s try and define a
 threshold to identify an outlier.

"""
threshold = 3
print(np.where(z > 3))

#interquartile range (IQR)## INTERQUARTILE RANGE............


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

"""
The below code will give an output with some true and false values.
The data point where we have False that means these values 
are valid whereas True indicates presence of an outlier.
"""

print(df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))



"""
Working with Outliers: Correcting, Removing

"""

df = df[(z < 3).all(axis=1)]
df.cardio.value_counts()
df.head()
df.ap_hi.value_counts()
df.describe()

df['smo_alc'] = df['smoke'] + df['alco']

7.#Cholesterol /| 1: normal, 2: above normal, 3: well above normal |
8.#Glucose /| 1: normal, 2: above normal, 3: well above normal |


df['chlo_glu'] = df['cholesterol'] + df['gluc']

df['all_SACG'] = df['smo_alc'] + df['chlo_glu']


        
"""
Body Mass Index Chart    .
Weight Status	Body Mass Index
Underweight	Below 18.5
Normal	18.5 to 24.9
Overweight	25.0 to 29.9
Obese	30.0 and Above
"""
#AS DATA IN FORM HAVING NO MISSING VALUE
#CHECKING THE OUTLIERS
df.info()
"""
EXPLORATORY DATA ANALYSIS

"""
#univariate and multi-variate analysis.
sns.countplot(df.BMI)

#
## 
###
####Analysing the variable AGE AND CARDIO

sns.FacetGrid(df,hue = 'cardio',size=10).map(sns.distplot,'age').add_legend()
df.info()
sns.FacetGrid(df,hue = 'cardio',size=10).map(sns.distplot,'all_SACG').add_legend()

sns.distplot(df['BMI'], kde=False)

plt.hist(df.age)

plt.boxplot(df['BMI'])
#Creating age group in AGE columns as AGING
#df.loc[(df['age'] <= 12), 'Aging'] = '0-12'
#df.loc[((df['age'] >= 12) & (df['age'] <= 24)) , 'Aging'] = '12-24'
#df.loc[((df['age'] >= 24) & (df['age'] <= 48)) , 'Aging'] = '22-48'
#df.loc[((df['age'] >= 48) & (df['age'] <= 60)) , 'Aging'] = '48-60'
#df.loc[((df['age'] >= 60) & (df['age'] <= 70)) , 'Aging'] = '60-70'
df.head(4)
# ***Where  | 1: normal, 2: above normal, 3: well above normal |

#sns.countplot(df.ap_hi)
df['ap_hi'].value_counts().head(10).plot.bar()
df['ap_hi'].plot.hist()

## Joint plots shows bivariate scatterplots
# And univariate histograms

sns.jointplot(x=df.age, y=df.cardio, kind="kde", data = df)

#plotting pairplot MULTIVARIATE ANALYSIS

sns.pairplot(df,hue = 'cardio', size =3) 

df.info()


#sns.boxplot(x='age', y='cardio', data=df)
#plt.show()
#multivaiate analysis


"""
VIF =Variation Inflation Factor

"""
# Import library for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif_(X):

    '''X - pandas dataframe'''
    thresh = 5.0
    variables = range(X.shape[1])

    for i in np.arange(0, len(variables)):
        vif = [variance_inflation_factor(X[variables].values, ix) for ix in range(X[variables].shape[1])]
        print(vif)
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X[variables].columns[maxloc] + '\' at index: ' + str(maxloc))
            del variables[maxloc]

    print('Remaining variables:')
    print(X.columns[variables])
    return X


corr = df.corr()
sns.set_context("notebook", font_scale=0.5, rc={"lines.linewidth": 2.5})
plt.figure(figsize=(14,10))
a = sns.heatmap(corr, annot=True,cmap='coolwarm')
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)


df.info()

#remove the high multicollinearity variable
df.drop(['height1'],axis=1, inplace = True) 
df.drop(['id'],axis=1, inplace = True)
df.drop(['cholesterol','gluc','smoke','alco','smo_alc','chlo_glu'],axis=1,inplace = True) 

#df.drop([''])

df.info()
#train and test splitting
from sklearn.model_selection import train_test_split
X = df.drop('cardio',axis=1)
y = df['cardio']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2,random_state=150)

# We'll need some metrics to evaluate our models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score


# Random Forest 
#--------------
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=250)
classifier.fit( X_train, y_train )
y_pred = classifier.predict( X_test )


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

cm = confusion_matrix( y_test, y_pred )
print("Accuracy on Test Set for RandomForest = %.2f" % ((cm[0,0] + cm[1,1] )/len(X_test)))
scoresRF = cross_val_score( classifier, X_train, y_train, cv=80)
print("Mean RandomForest CrossVal Accuracy on Train Set %.2f, with std=%.2f" % (scoresRF.mean(), scoresRF.std() ))

# Logistic Regression 
#--------------
from sklearn.linear_model import LogisticRegression
classifier2 = LogisticRegression()
classifier2.fit( X_train, y_train )
y_pred = classifier2.predict( X_test )


cm = confusion_matrix( y_test, y_pred )
print("Accuracy on Test Set for LogReg = %.2f" % ((cm[0,0] + cm[1,1] )/len(X_test)))
scoresLR = cross_val_score( classifier2, X_train, y_train, cv=10)
print("Mean LogReg CrossVal Accuracy on Train Set %.2f, with std=%.2f" % (scoresLR.mean(), scoresLR.std() ))













