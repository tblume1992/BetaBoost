# -*- coding: utf-8 -*-
r"""
Created on Mon Dec  2 22:51:33 2019
@author: Tyler Blume
"""
from scipy.stats import beta
import xgboost as xgb


class BetaBoost:
    """
    BetaBoost uses the idea that there may be a more 'useful' set of learning 
    rates to use while boosting than standard constant or the more rare decay
    rate.  Instead, we have noted some improvements while using a spike in 
    'adolescent trees' (trees in rounds 9-15 of boosting). To achieve this spike
    we use the Beta distribution function with some guardrails. To understand 
    why you can see the incoherent reasoning here! 

    https://github.com/tblume1992/portfolio/blob/master/GradientBoostedTrees/3_Dynamic_GBT.ipynb

    We are definitely NOT saying that this is the answer to everything.  Rather,
    it is quite interesting and more research should be done in the area.
    """
    __author__ = 'Tyler Blume'
    __version__ = '0.0.2'

    def __init__(self,
                 scalar=1.5,
                 a=26,
                 b=1,
                 scale=80,
                 loc=-68,
                 floor=0.01,
                 n_boosting_rounds=10
                 ):
        self.scalar = scalar
        self.a = a
        self.b = b
        self.scale = scale
        self.loc = loc
        self.floor = floor
        self.n_boosting_rounds = n_boosting_rounds

    def beta_kernel(self):
        """
        Get the 

        Parameters
        ----------
        Returns
        -------
        lrs : TYPE
            DESCRIPTION.

        """
        lrs = [self.scalar*beta.pdf(i,
                                    a=self.a, 
                                    b=self.b, 
                                    scale=self.scale, 
                                    loc=self.loc) 
               + self.floor for i in range(self.n_boosting_rounds)]
        return lrs
  
    def fit(self, **kwargs):
        """
        Fit the betaboost model which is simply XGBoost's train class that allows us to pass learning rates as a list.
        View more: https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.training
    
        Parameters
        ----------
        **kwargs : TYPE
            kwargs relating to XGBoost's train class.
          
        Returns
        -------
        xgb_train_obj : TYPE
            The output is the output from XGBoost's train class.
          
        """
        lrs = self.beta_kernel()
        xgb_train_obj = xgb.train(num_boost_round=self.n_boosting_rounds,
                                  learning_rates=lrs,
                                  **kwargs)
        return xgb_train_obj
