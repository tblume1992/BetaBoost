# BetaBoost
A small wrapper to do Beta Boosting with XgBoost

```pip install BetaBoost==0.0.4```

Initiate a BetaBoost object and fit a XgBoost model.
Returns a XgBoost Train Object.

Betaboosting assumes that there is some spike in the learning rate that exists which can give 'optimal' performance, a beta kernel with a floor is used for convenience.

A quick example with some toy data.  Found this example awhile ago for learning rate decay:

```python
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

xgb.__version__

def generate_data():
    y = np.random.gamma(2, 4, OBS)
    X = np.random.normal(5, 2, [OBS, FEATURES])
    return X, y

max_iter = 300
eta_base = 0.2
eta_min = 0.1
eta_decay = np.linspace(eta_base, eta_min, max_iter).tolist()
OBS = 10 ** 4
FEATURES = 20
PARAMS = {
    'eta': eta_base,
    "booster": "gbtree",
    "silient": 1,
}


X_train, y_train = generate_data()
X_test, y_test = generate_data()
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
evals_result = {'train': dtrain}

progress1 = dict()
model1 = xgb.train(
    maximize=True,
    params=PARAMS,
    dtrain=dtrain,
    num_boost_round=max_iter,
    early_stopping_rounds=max_iter,
    evals=[(dtrain, 'train'),(dtest, 'test')],
    evals_result=progress1,
    verbose_eval=False,
    callbacks=[xgb.callback.LearningRateScheduler(eta_decay)]
)

progress2 = dict()
model2 = xgb.train(
    maximize=True,
    params=PARAMS,
    dtrain=dtrain,
    num_boost_round=max_iter,
    early_stopping_rounds=max_iter,
    evals=[(dtrain, 'train'),(dtest, 'test')],
    evals_result=progress2,
    verbose_eval=False,
    callbacks=[xgb.callback.LearningRateScheduler(list(np.ones(max_iter)*0.01))]
)


progress3 = dict()
model3 = xgb.train(
    maximize=True,
    params=PARAMS,
    dtrain=dtrain,
    num_boost_round=max_iter,
    early_stopping_rounds=max_iter,
    evals=[(dtrain, 'train'),(dtest, 'test')],
    evals_result=progress3,
    verbose_eval=False,
    callbacks=[xgb.callback.LearningRateScheduler(list(np.ones(max_iter)*0.1))]
)

#Here we call the BetaBoost, the wrapper parameters are passed in the class init
bb_evals = dict()
import BetaBoost as bb
betabooster = bb.BetaBoost(n_boosting_rounds = max_iter)
betabooster.fit(dtrain=dtrain,
                maximize=True,
                params = PARAMS,
                early_stopping_rounds=max_iter,
                evals=[(dtrain, 'train'),(dtest, 'test')],
                evals_result = bb_evals,
                verbose_eval = False)

plt.plot(progress1['test']['rmse'], linestyle = 'dashed', color = 'b', label = 'eta test decay')
plt.plot(progress2['test']['rmse'], linestyle = 'dashed', color = 'r', label = '0.01 test')
plt.plot(progress3['test']['rmse'], linestyle = 'dashed', color = 'black', label = '0.1 test')
plt.plot(bb_evals['test']['rmse'], linestyle = 'dashed', color = 'y', label = 'bb test')
plt.legend()
plt.show()
```
![alt text](https://github.com/tblume1992/BetaBoost/blob/master/static/betaboost_test_image.png?raw=true "BB Awesome Image")

A clear nice-ety of beta boosting is that the model's test set is one of the first models to converge and yet is resistant to overfitting.  Plotting the default beta kernel settings will show a function that starts quite small and peaks around .5 by iteration 30.  This process allows the model to make the largest jumps not when the trees are super weak (standard eta decay) or when they are too strong.

A quick convergence and robustness to overfitting can be achieved with slight tunings of the beta kernel parameters...it's only 6 more parameters guys :)

FAQ:

Why a beta pdf?

TLDR: it worked pretty well!

Longer incohernt look at it: https://github.com/tblume1992/portfolio/blob/master/GradientBoostedTrees/3_Dynamic_GBT.ipynb
