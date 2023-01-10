# https://blog.csdn.net/luanfenlian0992/article/details/106448500

from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import datetime

data = load_boston()
X = data.data
y = data.target

Xtrain, Xtest, Ytrain, Ytest = TTS(X, y, test_size=0.3, random_state=420)
reg = XGBR(n_estimators=100).fit(Xtrain, Ytrain)
reg.predict(Xtest)
reg.score(Xtest, Ytest)

MSE(Ytest, reg.predict(Xtest))
print(reg.feature_importances_)

reg = XGBR(n_estimators=100)
CVS(reg, Xtrain, Ytrain, cv=5).mean()
CVS(reg, Xtrain, Ytrain, cv=5, scoring="neg_mean_squared_error").mean()

import sklearn
sorted(sklearn.metrics.SCORERS.keys())
rfr = RFR(n_estimators=100)
CVS(rfr, Xtrain, Ytrain, cv=5).mean()
CVS(rfr, Xtrain, Ytrain, cv=5, scoring="neg_mean_squared_error").mean()
lr = LinearR()
CVS(lr, Xtrain, Ytrain, cv=5).mean()
CVS(lr, Xtrain, Ytrain, cv=5, scoring="neg_mean_squared_error").mean()
reg = XGBR(n_estimators=10, silent=False)
CVS(reg, Xtrain, Ytrain, cv=5, scoring="neg_mean_squared_error").mean()


def plot_learning_curve(estimator, title, X, y, ax=None, ylim=None, cv=None, n_jobs=None):
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    import numpy as np
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, shuffle=True, cv=cv, n_jobs=n_jobs)
    if ax == None:
        ax = plt.gca()
    else:
        ax = plt.figure()
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.grid()
    ax.plot(train_sizes, np.mean(train_scores, axis=1), "o-", color="r", label="Training score")
    ax.plot(train_sizes, np.mean(train_scores, axis=1), "o-", color="g", label="Test score")
    ax.legend(loc="best")
    return ax

cv = KFold(n_splits=5, shuffle=True, random_state=42)
plot_learning_curve(XGBR(n_estimators=100, random_state=420), "XGB", Xtrain, Ytrain, ax=None, cv=cv)
plt.savefig("./xgb.png")
plt.show()


# axisx = range(10,1010,50)
# rs = []
# for i in axisx:
#     reg = XGBR(n_estimators=i,random_state=420)
# rs.append(CVS(reg,Xtrain,Ytrain,cv=cv).mean())
# print(axisx[rs.index(max(rs))],max(rs)) 
# plt.figure(figsize=(20,5)) 
# plt.plot(axisx,rs,c="red",label="XGB") 
# plt.legend()
# plt.savefig("xgb2.png")
# plt.show()

axisx = range(50,1050,50)
rs = []
var = []
ge = []
for i in axisx:
	reg = XGBR(n_estimators=i,random_state=420) 
	cvresult = CVS(reg,Xtrain,Ytrain,cv=cv) #记录1-偏差
	rs.append(cvresult.mean())
	#记录方差
	var.append(cvresult.var())
	#计算泛化误差的可控部分
	ge.append((1 - cvresult.mean())**2+cvresult.var())
#打印R2最高所对应的参数取值，并打印这个参数下的方差
print(axisx[rs.index(max(rs))],max(rs),var[rs.index(max(rs))]) 
#打印方差最低时对应的参数取值，并打印这个参数下的R2 
print(axisx[var.index(min(var))],rs[var.index(min(var))],min(var)) 
#打印泛化误差可控部分的参数取值，并打印这个参数下的R2，方差以及泛化误差的可控部分
print(axisx[ge.index(min(ge))],rs[ge.index(min(ge))],var[ge.index(min(ge))],min(ge))
plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="red",label="XGB")
plt.legend()
plt.savefig("xgb2.png")
plt.show()


axisx = range(100,300,10)
rs = []
var = []
ge = []
for i in axisx:
	reg = XGBR(n_estimators=i,random_state=420) 
	cvresult = CVS(reg,Xtrain,Ytrain,cv=cv) rs.append(cvresult.mean()) 	
	var.append(cvresult.var())
	ge.append((1 - cvresult.mean())**2+cvresult.var())
print(axisx[rs.index(max(rs))],max(rs),var[rs.index(max(rs))]) 
print(axisx[var.index(min(var))],rs[var.index(min(var))],min(var)) 
print(axisx[ge.index(min(ge))],rs[ge.index(min(ge))],var[ge.index(min(ge))],min(ge)) 
rs = np.array(rs)
var = np.array(var)*0.01 
plt.figure(figsize=(20,5)) 
plt.plot(axisx,rs,c="black",label="XGB") 
#添加方差线 
plt.plot(axisx,rs+var,c="red",linestyle='-.') 
plt.plot(axisx,rs-var,c="red",linestyle='-.') 
plt.legend()
plt.show()

#看看泛化误差的可控部分如何? 
plt.figure(figsize=(20,5)) 
plt.plot(axisx,ge,c="gray",linestyle='-.') 
plt.savefig("xgb3.png")
plt.show()