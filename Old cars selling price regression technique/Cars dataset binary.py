
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_excel(r"C:\Users\mukul\Downloads\Data_Train.xlsx")

#dataset['Name'].nunique()
#dataset['Location'].nunique()

dataset['Engine'] = dataset.Engine.str.split().str[0]
dataset['Mileage'] = dataset.Mileage.str.split().str[0]
dataset['Power'] = dataset.Power.str.split().str[0]

dataset['Mileage'] = pd.to_numeric(dataset['Mileage'], errors ='coerce')
dataset['Engine'] = pd.to_numeric(dataset['Engine'], errors ='coerce')
dataset['Power'] = pd.to_numeric(dataset['Power'], errors = 'coerce')

dataset.info()
from sklearn.impute import SimpleImputer
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
dataset['Engine_imputed'] = mode_imputer.fit_transform(dataset.Engine.values.reshape(-1, 1))
dataset['Power_imputed'] = mode_imputer.fit_transform(dataset.Power.values.reshape(-1, 1))
dataset['Mileage_imputed'] = mode_imputer.fit_transform(dataset.Mileage.values.reshape(-1, 1))
dataset['Seats_imputed'] = mode_imputer.fit_transform(dataset.Seats.values.reshape(-1, 1))

new_column = ['Name', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type',
       'Transmission', 'Owner_Type', 'Mileage_imputed', 'Engine_imputed', 'Power_imputed',
       'Seats_imputed', 'New_Price', 'Price']

dataset = dataset[new_column]
dataset.columns = ['Name', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type',
       'Transmission', 'Owner_Type', 'Mileage', 'Engine', 'Power',
       'Seats', 'New_Price', 'Price']

dataset.info()
dataset.drop('New_Price', axis = 1, inplace = True)
#dataset.info()

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
dataset['Name'] = LE.fit_transform(dataset.Name)
#dataset.info()


 
listt = dataset.Name.values.tolist()
new_list = []
for i in listt:
    binary = format(i, '016b')[5:]
    new_list.append(binary)

#X['B_Name'] = new_list
#X['B_Name'] = pd.to_numeric(X['B_Name'], errors = 'coerce')
#X.info()

#X.drop('Name', axis = 1, inplace = True)
#X.info()
ab = []
for j in new_list:
    split = [int(i) for i in str(j)]
    ab.append(split)
ab = np.matrix(ab)
df = pd.DataFrame(ab, columns = 'A B C D E F G H I J K '.split())

dataset.drop('Name', axis = 1, inplace = True)
dataset.drop('Engine', axis = 1, inplace = True)    
dataset = pd.concat([df, dataset], axis = 1)


# Treating outlierrs
Q1 = np.quantile(dataset.Kilometers_Driven.values, .25)
Q3 = np.quantile(dataset.Kilometers_Driven.values, .75)
minn = (Q3 - 1.5*(Q3-Q1))
maxx = (Q3 + 1.5*(Q3-Q1))
outliers_dataset = dataset[(dataset['Kilometers_Driven'] > maxx) | 
                    (dataset['Kilometers_Driven'] < minn)]
cond = dataset['Kilometers_Driven'].isin(outliers_dataset['Kilometers_Driven']) == True
dataset.drop(dataset[cond].index, inplace = True)
plt.boxplot('Kilometers_Driven', data = dataset)
plt.show()

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
X['Location'] = LE.fit_transform(X.Location)
X['Fuel_Type'] = LE.fit_transform(X.Fuel_Type)
X['Transmission'] = LE.fit_transform(X.Transmission)
X['Owner_Type'] = LE.fit_transform(X.Owner_Type)

X = pd.get_dummies(X, columns = ['Owner_Type', 'Transmission', 
                                     'Fuel_Type', 'Location'])

X.drop(['Owner_Type_3', 'Transmission_1', 'Fuel_Type_4', 'Location_10'], 
       axis = 1, inplace = True)


import pandas_profiling as pp
A = pp.ProfileReport(dataset)
A.to_file('Car_binary_full.html')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25)

"""# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = y_train.values.reshape(-1, 1)
y_train = sc_y.fit_transform(y_train)"""

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 900, random_state = 0)
regressor.fit(X_train, y_train)
# Predicting a new result
y_pred = regressor.predict(X_test)
print(regressor.score(X = X_test,  y = y_test))

regressor.get_params()

parameters=[{'random_state': [0, 1, 2, 3, 4, 5],
             'n_estimators': [900, 950, 1000]}]

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=regressor,
                             param_grid=parameters,
                             scoring='r2',
                             n_jobs=-1,cv=3)
grid_search.fit(X_train, y_train)
grid_search.best_params_
grid_search.best_score_
# Fitting Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
print(lin_reg.score(X = X_test, y = y_test))

from sklearn.metrics import r2_score
print(r2_score(y_test, lin_reg.predict(X_test)))




from sklearn.preprocessing import PolynomialFeatures
polynomial_scores = []
for degree in range(1, 4):
    
    poly_features = PolynomialFeatures(degree = degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.fit_transform(X_test)
    #poly_reg.fit(X_poly, y)
    polynomial_regressor = LinearRegression()
    polynomial_regressor.fit(X_train_poly, y_train)
    
    
    y_pred  = polynomial_regressor.predict(X_test_poly)
    #print(polynomial_regressor.score(X = X_test, y = y_test))
    
    polynomial_scores.append(r2_score(y_test, y_pred))

plt.plot(range(1,4), polynomial_scores)
plt.show()


'''
# Fitting Polynimial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 1)
X_poly = poly_reg.fit_transform(X_train)
#poly_reg.fit(X_poly, y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)
y_pred  = lin_reg_2.predict(X_test)
print(lin_reg_2.score(X = X_test, y = y_test))'''
# Importing test dataset
dataset_test = pd.read_excel(r"C:\Users\mukul\Downloads\Data_Test.xlsx")
dataset_test.info()

dataset_test['Name'] = LE.transform(dataset_test.Name)