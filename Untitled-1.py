import pandas as pd
import quandl , datetime, math
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn import model_selection
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style 
import pickle
style.use('ggplot')
df = quandl.get('WIKI/GOOGL')
df = df [['Adj. Open','Adj. High', 'Adj. Low', 'Adj. Close','Adj. Volume',]]
df['HL_PCT']= (df ['Adj. High']- df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change']= (df ['Adj. Close']- df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df [['Adj. Close','HL_PCT','Adj. Volume']]
forecast_col = 'Adj. Close'
df.fillna(-9999, inplace=True)

forecast_out = int(math.ceil(0.1*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

X = preprocessing.scale(X)
df.dropna(inplace=True)
Y = np.array(df['label'])


X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y, test_size=0.2)

#clf = LinearRegression(n_jobs=10)
#clf.fit(X_train, Y_train)
#with open('linearregression.pickle','wb') as f:
    #pickle.dump(clf, f)
    
pickle_in =open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)
accuracy = clf.score(X_test,Y_test)
#print(accuracy)
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)
df['forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date]= [np.nan for _ in range(len(df.column)-1) ]+[i]
df['Adj.Close'].plot()
df['forecast'].plot()
plot.legend(loc=4)
plt.Xlabel('date')
plt.Ylabel('price')
plt.show()