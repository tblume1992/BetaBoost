# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 22:51:33 2019

@author: TXB3Y48
"""
from scipy.stats import beta
import xgboost as xgb


class BetaBoost:
  def __init__(self,
               scalar = 1.5,
               a = 26,
               b = 1,
               scale = 80,
               loc = -68,
               floor = 0.01,
               n_boosting_rounds = 10
               ):
    self.scalar = scalar
    self.a = a
    self.b = b
    self.scale = scale
    self.loc = loc
    self.floor = floor
    self.n_boosting_rounds = n_boosting_rounds
    return

  def beta_kernel(self, length):
    scalar = self.scalar
    a = 26
    b = 1
    scale = 80
    loc = -68
    lrs = [scalar*beta.pdf(i, a=a, b=b, scale=scale, loc=loc) + 0.01 for i in range(length)]
    return lrs
  
  def fit(self,
          dtrain,
          dtest,
          params = dict(),
          evals_result = None,
          verbose_eval = False,
          ):
    
    bb = xgb.train(
    maximize=True,
    params=params,
    dtrain=dtrain,
    num_boost_round=self.n_boosting_rounds,
    evals=[(dtrain, 'train'),(dtest, 'test')],
    evals_result=evals_result,
    verbose_eval=verbose_eval,
    learning_rates=self.beta_kernel(self.n_boosting_rounds)
    )
    return bb
