# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 14:22:26 2017

@author: jpphooper
"""

import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib.pylab as pyl
import numpy as np
import statsmodels.api as sm
import scipy as sp
import random 
from tslearn.clustering import GlobalAlignmentKernelKMeans
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import sigma_gak, cdist_gak
from tslearn.datasets import CachedDatasets
import matplotlib.dates as dates
import datetime as dt1
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV
import plotly.plotly as py

dfCountryTemp = pd.read_csv('GlobalLandTemperaturesByCountry.csv', parse_dates=[0]) # Reading data
dfGlobalTemp = pd.read_csv('GlobalTemperatures.csv', index_col = 'dt', parse_dates=[0])
dfCityTemp = pd.read_csv('GlobalLandTemperaturesByMajorCity.csv', parse_dates=[0])

print(dfCountryTemp.head())
print (dfCountryTemp.describe()) # Calculating useful stats
print (dfCountryTemp.shape[0] - dfCountryTemp.count()) # Checking for missing data

plt.figure()
pd.rolling_mean(dfGlobalTemp[dfGlobalTemp.index.year > 1900]['LandAverageTemperature'],window=60).plot(x=dfGlobalTemp.index)
pd.rolling_mean(dfGlobalTemp[dfGlobalTemp.index.year > 1900]['LandMaxTemperature'],window=60).plot(x=dfGlobalTemp.index)
pd.rolling_mean(dfGlobalTemp[dfGlobalTemp.index.year > 1900]['LandMinTemperature'],window=60).plot(x=dfGlobalTemp.index)
plt.show()


dfCountryTempPivot = dfCountryTemp.pivot(index='dt', columns='Country', values='AverageTemperature')
dfCityTempPivot = dfCityTemp.pivot(index='dt', columns='City', values='AverageTemperature')

df = pd.DataFrame(dfCountryTempPivot)

# Let's use some interpolation methods to fill in blank values

df = df.interpolate(method='ffill')
df = df.dropna()
df1 = df
df = pd.rolling_mean(df,window=60)
df = df[df.index.year > 1900]
df = df.dropna()
df = df.iloc[1:1351].subtract(df.iloc[0])
df1 = df1.dropna()
df1 = df1.iloc[1:1351].subtract(df.iloc[0])
df = df.drop('French Southern And Antarctic Lands', 1)
df = df.drop('Heard Island And Mcdonald Islands', 1)
df = df.drop('Federated States Of Micronesia', 1)
df = df.drop('Northern Mariana Islands', 1)
df = df.drop('Palau', 1)
df1 = df1.drop('French Southern And Antarctic Lands', 1)
df1 = df1.drop('Heard Island And Mcdonald Islands', 1)
df1 = df1.drop('Federated States Of Micronesia', 1)
df1 = df1.drop('Northern Mariana Islands', 1)
df1 = df1.drop('Palau', 1)
output = df.index.map(str)
x = [dt1.datetime.strptime(d,'%Y-%m-%d %H:%M:%S').date() for d in output]



TSValues = [df.values]
TSValues = np.transpose(TSValues)
TSValues1 = [df1.values]
TSValues1 = np.transpose(TSValues1)

from tslearn.utils import to_time_series_dataset

formatted_dataset = to_time_series_dataset(TSValues)
formatted_dataset1 = to_time_series_dataset(TSValues1)

print(formatted_dataset.shape)

from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
from tslearn.neighbors import KNeighborsTimeSeries


seed=0
np.random.seed(seed)
#np.random.shuffle(formatted_dataset)
#formatted_dataset = TimeSeriesScalerMeanVariance().fit_transform(formatted_dataset) 
#formatted_dataset = TimeSeriesResampler(sz=1000).fit_transform(formatted_dataset)
sz = formatted_dataset.shape[1]
#km = GlobalAlignmentKernelKMeans(n_clusters=3, sigma=sigma_gak(formatted_dataset), n_init=20, verbose=True, random_state=seed)
km = TimeSeriesKMeans(n_clusters=6, verbose=True, random_state=seed)
km1 = TimeSeriesKMeans(n_clusters=6, verbose=True, random_state=seed)
y_pred = km.fit_predict(formatted_dataset)
y_pred1 = km1.fit_predict(formatted_dataset1)
print(y_pred)

plt.figure(figsize=(7,9))
for yi in range(2):
    plt.subplot(3, 2, yi + 1)
    for xx in formatted_dataset[y_pred == yi]:
        plt.plot(x, xx.ravel(), "k-", alpha=.2)
    if yi == 0:
        plt.plot(x, km.cluster_centers_[yi].ravel(), "b-")
    else:
        plt.plot(x, km.cluster_centers_[yi].ravel(), "-", color = 'orange')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(dates.DateFormatter("%Y-%m"))
    #ax.xaxis.set_major_locator(dates.DayLocator())
    _=plt.xticks(rotation=90)
    pyl.title("Cluster %s"%(yi))

for yi in range(2, 4):
    plt.subplot(3, 2, yi + 1)
    for xx in formatted_dataset[y_pred == yi]:
        plt.plot(x, xx.ravel(), "k-", alpha=.2)
    if yi == 2:
        plt.plot(x, km.cluster_centers_[yi].ravel(), "r-")
    else:
        plt.plot(x, km.cluster_centers_[yi].ravel(), "c-")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(dates.DateFormatter("%Y-%m"))
    _=plt.xticks(rotation=90)
    pyl.title("Cluster %s"%(yi))
    if yi == 2 :
        plt.ylabel('Temperature Change since 1900 (degrees C)')

for yi in range(4, 6):
    plt.subplot(3, 2, yi + 1)
    for xx in formatted_dataset[y_pred == yi]:
        plt.plot(x, xx.ravel(), "k-", alpha=.2)
    if yi == 4:
        plt.plot(x, km.cluster_centers_[yi].ravel(), "g-")
    else:
        plt.plot(x, km.cluster_centers_[yi].ravel(), "y-")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(dates.DateFormatter("%Y-%m"))
    _=plt.xticks(rotation=90)
    pyl.title("Cluster %s"%(yi))

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.suptitle("Euclidean $k$-means", fontsize=16)
plt.savefig('TimeSeriesClusters.eps', format='eps', dpi=1000)
plt.show()

Countries = (df.columns.astype(str).tolist())
Countries = np.asarray(Countries, dtype='str')
CountryClusters = np.array([[d, c] for d, c in zip(Countries, y_pred)])

Cluster0Centre = km.cluster_centers_[0]
Cluster1Centre = km.cluster_centers_[1]
Cluster2Centre = km.cluster_centers_[2]
Cluster3Centre = km.cluster_centers_[3]
Cluster4Centre = km.cluster_centers_[4]
Cluster5Centre = km.cluster_centers_[5]

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.8)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error
 
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    return best_cfg
"""
p_values = [0, 1, 2, 4]
d_values = range(0, 2)
q_values = range(0, 2)
warnings.filterwarnings("ignore")
evaluate_models(Cluster0Centre, p_values, d_values, q_values)
"""
warnings.filterwarnings("ignore")
model0 = ARIMA(Cluster0Centre, (12,0,1))
model0_fit = model0.fit(disp=0)
model0forecast = model0_fit.forecast(steps=13)[0]
model0forecast = model0forecast - model0forecast[0]
model1 = ARIMA(Cluster1Centre, (12,0,1))
model1_fit = model1.fit(disp=0)
model1forecast = model1_fit.forecast(steps=13)[0]
model1forecast = model1forecast - model1forecast[0]
model2 = ARIMA(Cluster2Centre, (12,0,1))
model2_fit = model2.fit(disp=0)
model2forecast = model2_fit.forecast(steps=13)[0]
model2forecast = model2forecast - model2forecast[0]
model3 = ARIMA(Cluster3Centre, (12,0,1))
model3_fit = model3.fit(disp=0)
model3forecast = model3_fit.forecast(steps=13)[0]
model3forecast = model3forecast - model3forecast[0]
model4 = ARIMA(Cluster4Centre, (12,0,1))
model4_fit = model4.fit(disp=0)
model4forecast = model4_fit.forecast(steps=13)[0]
model4forecast = model4forecast - model4forecast[0]
model5 = ARIMA(Cluster5Centre, (12,0,1))
model5_fit = model5.fit(disp=0)
model5forecast = model5_fit.forecast(steps=13)[0]
model5forecast = model5forecast - model5forecast[0]

plt.figure()
plt.plot(model0forecast, 'b-', label = 'Cluster 0')
plt.plot(model1forecast, '-', color = 'orange', label = 'Cluster 1')
plt.plot(model2forecast, 'r-', label = 'Cluster 2')
plt.plot(model3forecast, 'c-',  label = 'Cluster 3')
plt.plot(model4forecast, 'g-',  label = 'Cluster 4')
plt.plot(model5forecast, 'y-', label = 'Cluster 5')
plt.xlabel('Months')
plt.ylabel('Temperature Change since before Forecast (Degrees C)')
plt.title('12 Month ARIMA Forecast')
plt.legend(loc=3)
plt.savefig('ARIMA.eps', format='eps', dpi=1000)
plt.show()



dfIndicators = pd.read_csv('Indicators.csv')
dfIndicators14 = dfIndicators[(dfIndicators['Year'] == 2014)]
dfIndicators13 = dfIndicators[(dfIndicators['Year'] == 2013)]
dfIndicators12 = dfIndicators[(dfIndicators['Year'] == 2012)]
dfIndicators11 = dfIndicators[(dfIndicators['Year'] == 2011)]
dfIndicators10 = dfIndicators[(dfIndicators['Year'] == 2010)]
dfIndicators09 = dfIndicators[(dfIndicators['Year'] == 2009)]
#dfIndicators.set_index([ 'Year','CountryName'], append=True)
dfInd14 = dfIndicators13.pivot(index = 'CountryName', columns = 'IndicatorName', values = 'Value')
dfInd14['Year'] = 2014
dfInd13 = dfIndicators13.pivot(index = 'CountryName', columns = 'IndicatorName', values = 'Value')
dfInd13['Year'] = 2013
dfInd12 = dfIndicators12.pivot(index = 'CountryName', columns = 'IndicatorName', values = 'Value')
dfInd12['Year'] = 2012
dfInd11 = dfIndicators11.pivot(index = 'CountryName', columns = 'IndicatorName', values = 'Value')
dfInd11['Year'] = 2011
dfInd10 = dfIndicators10.pivot(index = 'CountryName', columns = 'IndicatorName', values = 'Value')
dfInd10['Year'] = 2010
dfInd09 = dfIndicators09.pivot(index = 'CountryName', columns = 'IndicatorName', values = 'Value')
dfInd09['Year'] = 2009
dfClusters = pd.DataFrame(CountryClusters,index=CountryClusters[:,0])
dfClusters.to_csv('Clusters2013.csv')

dfInd = dfInd09.append([dfInd10,dfInd11,dfInd12,dfInd13,dfInd14])

dfInd2 = pd.merge(dfInd, dfClusters, how='inner', left_index = True,right_index = True)
dfInd2 = dfInd2.dropna(axis=1, thresh = 600)
dfInd2 = dfInd2.dropna(axis=0, thresh = 300)
dfInd2 = dfInd2.fillna(0)
XPCA = dfInd2.ix[:,1:535].values
Country = dfInd2.index.values
Country.shape = (900,1)
Years = dfInd2.ix[:,536].values
Years.shape = (900,1)
YPCA = dfInd2.ix[:,538].values
YPCA.shape = (900,1)


X_std = StandardScaler().fit_transform(XPCA)
sklearn_pca = sklearnPCA(n_components=20)
Y_sklearn = sklearn_pca.fit_transform(X_std)
Y_sklearn = np.append(Y_sklearn, Years, 1)
Y_sklearn = np.append(Y_sklearn, YPCA, 1)
Y_sklearn = np.append(Y_sklearn, Country, 1)

"""
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for lab, col in zip((0, 1, 2, 3, 4, 5),
                        ('blue', 'red', 'green', 'yellow', 'orange', 'black')):
        plt.scatter(Y_sklearn[YPCA==lab, 0],
                    Y_sklearn[YPCA==lab, 1],
                    label=lab,
                    c=col)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()
"""

print(sklearn_pca.explained_variance_ratio_[:])

Y_sklearn2014 = Y_sklearn[Y_sklearn[:,20] == 2014]
Y_sklearn = Y_sklearn[Y_sklearn[:,20] != 2014]
YPCA = Y_sklearn[:,21]
YPCA2014 = Y_sklearn2014[:,21]
Y_sklearn = Y_sklearn[:,0:19]
Country2014 = Y_sklearn2014[:,22]
Country2014.shape = (148,1)
Y_sklearn2014 = Y_sklearn2014[:,0:19]

train_x, test_x, train_y, test_y = train_test_split(Y_sklearn, YPCA, train_size=0.8)

print("Train_x Shape :: ", train_x.shape)
print("Train_y Shape :: ", train_y.shape)
print("Test_x Shape :: ", test_x.shape)
print("Test_y Shape :: ", test_y.shape)

clf = RandomForestClassifier()
clf.fit(train_x, train_y)
"""
param_grid = {'n_estimators': range(1,500)}

warnings.filterwarnings("ignore")

rsearch = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, n_iter=100)
rsearch.fit(train_x, train_y)

print(rsearch)
"""
predictions = clf.predict(test_x)

for i in range(0, 5):
        print("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]))

print("Train Accuracy :: ", accuracy_score(train_y, clf.predict(train_x)))
print("Test Accuracy  :: ", accuracy_score(test_y, predictions))
print("Confusion matrix ", confusion_matrix(test_y, predictions))

predictions2014 = clf.predict(Y_sklearn2014)
predictions2014.shape = (148,1)
Map2014 = np.append(Country2014,predictions2014,1)
dfMap = pd.DataFrame(Map2014)
dfMap.to_csv('Clusters2014.csv')

"""
data = [ dict(
        type = 'choropleth',
        locations = dfMap[0],
        z = dfMap[1],
        text = dfMap[0],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '$',
            title = 'Cluster Class'),
      ) ]

layout = dict(
    title = '2014 Global GDP<br>Source:\
            <a href="https://www.cia.gov/library/publications/the-world-factbook/fields/2195.html">\
            CIA World Factbook</a>',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='d3-world-map' )
"""