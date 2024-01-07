# Data Mining - cluster analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
desired_width = 400
pd.set_option('display.width', desired_width)         # sets run screen width to 400
pd.set_option('display.max_columns', 20)              # sets run screen column display to 20
df = pd.read_csv(r'single_family_home_values.csv')    # reads Zillow file
df2 = df.drop('estimated_value', axis = 1)            # any data frame column can be dropped
df3 = df2[['bedrooms', 'bathrooms', 'rooms', 'squareFootage', 'lotSize', 'yearBuilt', 'priorSaleAmount']]   # reduced df
df3.fillna(0, inplace=True)       # replaces the NaN in priorSaleAmount with 0 -- may get a warning, but better than NaN
print(df3.head(2))                                        # prints top two rows of df3
k_groups = KMeans(n_clusters=5, random_state=0).fit(df3)  # separates data set into 5 distinguishable groups
print(k_groups.labels_)                                   # displays k_groups' label (0 to 4) for each row
print(len(k_groups.labels_),df3.shape)                    # displays rows in k_groups as well as rows, columns in df3
print(k_groups.cluster_centers_)         # displays averages of the seven columns for each cluster centroid [0, 1, 2, 3, 4]
print(k_groups.cluster_centers_[0])       # displays averages for each of the seven columns in the cluster centroid [0]
df3['cluster'] = k_groups.labels_                         # add a new column to df3 called 'cluster', the k-group #
print(df3.head(3))                                        # display the top three rows of data frame df3
print(df3.groupby('cluster').mean())                      # display the means of the seven columns of data frame df3
from sklearn.metrics import silhouette_score              # coefficient score where higher is better, 0 = cluster overlap
df4 = df3.drop('cluster', axis = 1)                       # create a new data frame df4 that dropped the cluster column
# for loop to determine optimum K groups
for i in range(3, 10):                                    # for loop to determine best number of K clusters between 3 and 10
    k_groups = KMeans(n_clusters = i).fit(df4)            # K clusters must have atleast 2 clusters
    labels = k_groups.labels_
    print('K Groups = ', i, 'Silhouette Coeffient = ', silhouette_score(df4, labels))  # displays i and coefficient
# End of Data Mining - Cluster Analysis

# Data Mining - classification & regression
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
desired_width = 400
pd.set_option('display.width', desired_width)        # sets run screen width to 400
pd.set_option('display.max_columns', 20)             # sets run screen column display to 20
df = pd.read_csv(r'single_family_home_values.csv')   # reads Zillow file
df.fillna(0, inplace = True)                         # replaces the NaN with 0 to have even 15,000 in all 7 variables
X = df[['bedrooms', 'bathrooms', 'rooms', 'squareFootage', 'lotSize', 'yearBuilt', 'priorSaleAmount']] # reduced df as upper case X matrix
y = df.estimated_value                                           # predictor variable lower case y as array
print(type(X),type(y))
print(X.info())
print(X.shape, y.shape)

# Data Mining - classification & regression - LinearRegression()
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
desired_width = 400
pd.set_option('display.width', desired_width)        # sets run screen width to 400
pd.set_option('display.max_columns', 20)             # sets run screen column display to 20
df = pd.read_csv(r'single_family_home_values.csv')   # reads Zillow file
df.fillna(0, inplace = True)                         # replaces the NaN with 0 to have even 15,000 in all 7 variables
X = df[['bedrooms', 'bathrooms', 'rooms', 'squareFootage', 'lotSize', 'yearBuilt', 'priorSaleAmount']] # reduced df as upper case X matrix
y = df.estimated_value                                           # predictor variable lower case y as array
lg = LinearRegression()                                          # assigning alias lg to LinearRegression() function
print(lg.fit(X,y))                                               # training the regression model
print(lg.score(X,y))                                             # test the regression model
X_train, X_test, y_train, y_test = train_test_split(X,y)         # randomly split X,y data to 2 X,y (train,test) sets
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)  # we used default settings 80% is train to 20% test
print(lg.fit(X_train, y_train))                                  # Using 11,250 data points to train model
print(lg.score(X_test, y_test))                                  # Using  3,750 data points to test/evaluate R2 of model

# Data Mining - classification & regression - Logistic Regression
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
desired_width = 400
pd.set_option('display.width', desired_width)        # sets run screen width to 400
pd.set_option('display.max_columns', 20)             # sets run screen column display to 20
df = pd.read_csv(r'single_family_home_values.csv')   # reads Zillow file
df.fillna(0, inplace = True)                         # replaces the NaN with 0 to have even 15,000 in all 7 variables
X = df[['bedrooms', 'bathrooms', 'rooms', 'squareFootage', 'lotSize', 'yearBuilt', 'priorSaleAmount']] # reduced df as upper case X matrix
y = df.estimated_value                                           # predictor variable lower case y as array
df['estimated_value_class'] = df.estimated_value.apply(lambda x: 'low' if x < 500000 else 'high')
print(df.estimated_value_class.value_counts())                     # displays distribution of estimated_value_class
y2 = df.estimated_value_class                                      # assigns y2 as cat variable estimated_value_class
log = LogisticRegression()                                         # assigning alias lg to LogisticRegression() function
print(log.fit(X,y2))                                               # training the logistic regression model
print(log.score(X,y2))                                             # test the logistic regression model
X_train, X_test, y2_train, y2_test = train_test_split(X,y2)        # randomly split X,y data to 2 X,y (train,test) sets
print(X_train.shape, y2_train.shape, X_test.shape, y2_test.shape)  # we used default settings 80% is train to 20% test
print(log.fit(X_train, y2_train))                                    # Using 11,250 data points to train model
print(log.score(X_test, y2_test))                                    # Using  3,750 data points to test/evaluate R2 of model

# Data Mining - classification & regression - Prediction (Logistic)
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
desired_width = 400
pd.set_option('display.width', desired_width)        # sets run screen width to 400
pd.set_option('display.max_columns', 20)             # sets run screen column display to 20
df = pd.read_csv(r'single_family_home_values.csv')   # reads Zillow file
df.fillna(0, inplace = True)                         # replaces the NaN with 0 to have even 15,000 in all 7 variables
X = df[['bedrooms', 'bathrooms', 'rooms', 'squareFootage', 'lotSize', 'yearBuilt', 'priorSaleAmount']] # reduced df as upper case X matrix
y = df.estimated_value                               # predictor variable lower case y as array
df['estimated_value_class'] = df.estimated_value.apply(lambda x: 'low' if x < 500000 else 'high')
y2 = df.estimated_value_class                                      # assigns y2 as cat variable estimated_value_class
log = LogisticRegression()                                         # assigning alias lg to LogisticRegression() function
X_train, X_test, y2_train, y2_test = train_test_split(X,y2)        # randomly split X,y data to 2 X,y (train,test) sets
print(X_train.shape, y2_train.shape, X_test.shape, y2_test.shape)  # we used default settings 80% is train to 20% test
print(log.fit(X_train, y2_train))              # displays R2 from 11,250 data points to train model
print(log.score(X_test, y2_test))              # displays R2 from 3,750 data points used in 11,250 trained model
y2_pred = log.predict(X_test)                  # assigns y2_pred to prediction of X_test data
print(y2_pred, np.array(y2_test))              # displays y2_pred versus y2 of test data set
print(confusion_matrix(y2_test, y2_pred))      # displays correct on diagonal, typeI right, typeII left, all 3,750

# Data Mining - classification & regression - Prediction (Linear)
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
desired_width = 400
pd.set_option('display.width', desired_width)        # sets run screen width to 400
pd.set_option('display.max_columns', 20)             # sets run screen column display to 20
df = pd.read_csv(r'single_family_home_values.csv')   # reads Zillow file
df.fillna(0, inplace = True)                         # replaces the NaN with 0 to have even 15,000 in all 7 variables
X = df[['bedrooms', 'bathrooms', 'rooms', 'squareFootage', 'lotSize', 'yearBuilt', 'priorSaleAmount']] # reduced df as upper case X matrix
y = df.estimated_value                               # predictor variable lower case y as array
lg = LinearRegression()                              # assigning alias lg to LinearRegression() function
X_train, X_test, y_train, y_test = train_test_split(X,y)  # randomly split X,y data to 2 X,y (train,test) sets
lg.fit(X_train, y_train)                     # use 11,250 data points to train model
print(lg.score(X_test, y_test))              # displays R2 measure from 3,750 data points used in 11,250 trained model
y_prime = lg.predict(X_test)                 # assigns y_pred to prediction of X_test data
print(y_prime, np.array(y_test))             # displays y_pred versus y of test data set
print('MAE = ', mean_absolute_error(y_test, y_prime))  # displays mean absolute error of actual vs. predicted
print('MSE =', mean_squared_error(y_test, y_prime))   # displays mean squared error of actual vs. predicted
print('RMSE = ', np.sqrt(mean_squared_error(y_test, y_prime)))  # displays root mean squared error of actual vs. predicted

# Data Mining - classification & regression - SVR, SVC, Prediction Check
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
desired_width = 400
pd.set_option('display.width', desired_width)        # sets run screen width to 400
pd.set_option('display.max_columns', 20)             # sets run screen column display to 20
df = pd.read_csv(r'single_family_home_values.csv')   # reads Zillow file
df.fillna(0, inplace = True)                         # replaces the NaN with 0 to have even 15,000 in all 7 variables
X = df[['bedrooms', 'bathrooms', 'rooms', 'squareFootage', 'lotSize', 'yearBuilt', 'priorSaleAmount']] # reduced df as upper case X matrix
X1 = X                                               # duplicate of X
y = df.estimated_value                               # predictor variable lower case y as array
df['estimated_value_class'] = df.estimated_value.apply(lambda x: 'low' if x < 500000 else 'high')
y2 = df.estimated_value_class                                      # assigns y2 as cat variable estimated_value_class
X_train, X_test, y_train, y_test = train_test_split(X,y)        # randomly split X,y data to 2 X,y (train,test) sets
X1_train, X1_test, y2_train, y2_test = train_test_split(X1,y2)        # randomly split X,y data to 2 X,y (train,test) sets
svr_reg = SVR()                                          # assign svr_reg to the SVR function
svc_class = SVC()                                        # assign svc_class to the SVC function
svr_reg.fit(X_train, y_train)                            # fit a SVR() model using 11,250 data points
print('svr_score = ', svr_reg.score(X_test, y_test)) # score the SVR model based on 11,250 data points using the 3,750 data points
svc_class.fit(X1_train, y2_train)                        # fit a SVC() model using 11,250 data points
print('svc_score = ', svc_class.score(X1_test, y2_test)) # score the SVC model base on 11,250 data points using the 3,750 data points
y2_pred = svc_class.predict(X_test)                      # assigns y2_pred to SVC prediction of X_test data
print(confusion_matrix(y2_test, y2_pred))                # displays correct on diagonal, typeI right, typeII left, all 3,750

# Data Mining - classification & regression - K Nearest Neighbors Regressor & Classifier
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
desired_width = 400
pd.set_option('display.width', desired_width)        # sets run screen width to 400
pd.set_option('display.max_columns', 20)             # sets run screen column display to 20
df = pd.read_csv(r'single_family_home_values.csv')   # reads Zillow file
df.fillna(0, inplace = True)                         # replaces the NaN with 0 to have even 15,000 in all 7 variables
X = df[['bedrooms', 'bathrooms', 'rooms', 'squareFootage', 'lotSize', 'yearBuilt', 'priorSaleAmount']] # reduced df as upper case X matrix
X1 = X                                               # duplicate of X
y = df.estimated_value                               # predictor variable lower case y as array
df['estimated_value_class'] = df.estimated_value.apply(lambda x: 'low' if x < 500000 else 'high')
y2 = df.estimated_value_class                                      # assigns y2 as cat variable estimated_value_class
X_train, X_test, y_train, y_test = train_test_split(X,y)        # randomly split X,y data to 2 X,y (train,test) sets
X1_train, X1_test, y2_train, y2_test = train_test_split(X1,y2)        # randomly split X,y data to 2 X,y (train,test) sets
knn_reg = KNeighborsRegressor()                          # assign knn_reg to the KNeighborRegressor() function
knn_class = KNeighborsClassifier()                       # assign knn_class to the KNeighborClassifier function
knn_reg.fit(X_train, y_train)                            # fit a knn_reg model using 11,250 data points
print('knn_reg score = ', knn_reg.score(X_test, y_test)) # score the knn_reg model based on 11,250 data points using the 3,750 data points
knn_class.fit(X1_train, y2_train)                        # fit a knn_class model using 11,250 data points
print('knn_class score = ', knn_class.score(X1_test, y2_test)) # score the knn_class model base on 11,250 data points using the 3,750 data points
y2_pred = knn_class.predict(X_test)                      # assigns y2_pred to knn_class prediction of X_test data
print(confusion_matrix(y2_test, y2_pred))                # displays correct on diagonal, typeI right, typeII left, all 3,750
# End of Data Mining - Regression & Classification

# Data Mining - association & correlation
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
desired_width = 400
pd.set_option('display.width', desired_width)        # sets run screen width to 400
pd.set_option('display.max_columns', 20)             # sets run screen column display to 20
df = pd.read_csv(r'single_family_home_values.csv')   # reads Zillow file
df.fillna(0, inplace = True)                         # replaces the NaN with 0 to have even 15,000 in all 7 variables
X = df[['bedrooms', 'bathrooms', 'rooms', 'squareFootage', 'lotSize', 'yearBuilt', 'priorSaleAmount']] # reduced df as upper case X matrix
y = df.estimated_value                                           # predictor variable lower case y as array
print(df.corr())
sns.heatmap(df.corr())
plt.show()
print(df.cov())

# Data Mining - association & correlation - Histograms - outliers
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
desired_width = 400
pd.set_option('display.width', desired_width)        # sets run screen width to 400
pd.set_option('display.max_columns', 20)             # sets run screen column display to 20
df = pd.read_csv(r'single_family_home_values.csv')   # reads Zillow file
df.fillna(0, inplace = True)                         # replaces the NaN with 0 to have even 15,000 in all 7 variables
X = df[['bedrooms', 'bathrooms', 'rooms', 'squareFootage', 'lotSize', 'yearBuilt', 'priorSaleAmount']] # reduced df as upper case X matrix
y = df.estimated_value                               # predictor variable lower case y as array
print(X.columns)
for i in X.columns:
    print('     ')
    print(i)
    X.loc[:, i].hist()
    print('mean:  ', X.loc[:, i].mean())
    print('std:  ', X.loc[:, i].std())
    plt.show()

# Data Mining - association & correlation - Boxplots - outliers
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
desired_width = 400
pd.set_option('display.width', desired_width)        # sets run screen width to 400
pd.set_option('display.max_columns', 20)             # sets run screen column display to 20
df = pd.read_csv(r'single_family_home_values.csv')   # reads Zillow file
df.fillna(0, inplace = True)                         # replaces the NaN with 0 to have even 15,000 in all 7 variables
X = df[['bedrooms', 'bathrooms', 'rooms', 'squareFootage', 'lotSize', 'yearBuilt', 'priorSaleAmount']] # reduced df as upper case X matrix
y = df.estimated_value                               # predictor variable lower case y as array
sns.boxplot(X['bedrooms'])
plt.show()
sns.boxplot(X['bathrooms'])
plt.show()
sns.boxplot(X['rooms'])
plt.show()

# Data Mining - dimensionality reduction - Princple Component Analysis
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
desired_width = 400
pd.set_option('display.width', desired_width)        # sets run screen width to 400
pd.set_option('display.max_columns', 20)             # sets run screen column display to 20
df = pd.read_csv(r'single_family_home_values.csv')   # reads Zillow file
df.fillna(0, inplace = True)                         # replaces the NaN with 0 to have even 15,000 in all 7 variables
X = df[['bedrooms', 'bathrooms', 'rooms', 'squareFootage', 'lotSize', 'yearBuilt', 'priorSaleAmount']] # reduced df as upper case X matrix
X1 = X
pca = PCA(4)
X_transformed = pca.fit_transform(X)
y = df.estimated_value                               # predictor variable lower case y as array
y1 = df.estimated_value
lg = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y)        # randomly split X,y data to 2 X,y (train,test) sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1)        # randomly split X,y data to 2 X,y (train,test) sets
print(X_transformed.shape)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)  # we used default settings 80% is train to 20% test
lg.fit(X_train, y_train)                                         # Using 11,250 data points to train model
print('PCA = ', lg.score(X_test, y_test))                        # Using  3,750 data points to test/evaluate R2 of model
lg.fit(X1_train,y1_train)                                        # Using 11,250 data points to train model
print('non-PCA = ', lg.score(X1_test, y1_test))                  # Using  3,750 data points to test/evaluate R2 of model