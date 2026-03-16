import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
import statsmodels.api as sm
from itertools import combinations
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler

#===================================================================================================================================================#

class MeanCentering(BaseEstimator,TransformerMixin):
    def __init__(self):
      self.__name__='MeanCentering'
      self.mean=None
      self.fitted_ = False

    def fit(self,X,y=None):
      self.fitted_ = True
      try:
        X=pd.DataFrame(X)
      except:
        pass
      self.mean=X.mean(axis=0)
      return self
    def transform(self,X, y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      return pd.DataFrame(np.asarray(X)-np.asarray(self.mean))
    def fit_transform(self,X,y=None):
      self.fit(X)
      return self.transform(X) 

#===================================================================================================================================================#

class SNV(BaseEstimator,TransformerMixin):
    def __init__(self):
      self.__name__='SNV'

    def fit(self,X):
      pass
    def transform(self,X, y=None):
      try:
        X=pd.DataFrame(X)
        if X.shape[1] == 1:
          X=X.T
      except:
        pass
      R=(X.subtract(X.mean(axis=1), axis=0)).divide(X.std(axis=1)+np.finfo(float).eps, axis=0)
      return R
    def fit_transform(self,X,y=None):
      try:
        X=pd.DataFrame(X)
        if X.shape[1] == 1:
          X=X.T
      except:
        pass
      self.fit(X)
      return self.transform(X)
    
#===================================================================================================================================================#

class BaselineOffset(BaseEstimator, TransformerMixin):
    """Remove baseline offset from the data
    """
    def __init__(self):
        self.fitted_ = False
    def fit(self, X, y=None):
        """Not used
        """
        self.fitted_ = True
    def transform(self, X, y=None):
        """Removes baseline offset from data
        Args:
            X (Dataframe): Dataframe of samples [samples x variables] used to train pls_model
            y (_type_, optional): Not used. Defaults to None.
        Returns:
            Dataframe: X with baseline offset removed
        """
        X_new = X.sub(X.min(axis=1), axis=0)
        return X_new
    def fit_transform(self, X, y=None):
        """Removes baseline offset from data
        Args:
            X (Dataframe): Dataframe of samples [samples x variables] used to train pls_model
            y (_type_, optional): Not used. Defaults to None.
        Returns:
            Dataframe: X with baseline offset removed
        """
        return self.transform(X)

#===================================================================================================================================================#

class BaselineLinear(BaseEstimator, TransformerMixin):
    """Removes linear baseline
    """
    def __init__(self):
        """Not used
        """
        self.fitted_ = False
    def fit(self, X, y=None):
        """Not used
        """
        self.fitted_ = True
    def transform(self, X, y=None):
        """Removes linear baseline
        Args:
            X (Dataframe): Dataframe of samples [samples x variables] used to train pls_model
            y (_type_, optional): Not used. Defaults to None.
        Returns:
            Dataframe: X with linear baseline removed
        """
        X_new=X.copy()
        grid = np.arange(X.shape[1])
        for i in range(len(X_new)):
            y0, y1 = X.iloc[i,0], X.iloc[i,-1]
            grid0, grid1 = grid[0], grid[-1]
            # Compute the linear baseline between endpoints
            baseline = y0 + (grid - grid0) * (y1 - y0) / (grid1 - grid0)
            X_new.iloc[i,:] = X.iloc[i,:].sub(baseline)            
        return X_new
    def fit_transform(self, X, y=None):
        """Removes linear baseline
        Args:
            X (Dataframe): Dataframe of samples [samples x variables] used to train pls_model
            y (_type_, optional): Not used. Defaults to None.
        Returns:
            Dataframe: X with linear baseline removed
        """
        return self.transform(X)

#===================================================================================================================================================#

class FirstDerivative(BaseEstimator, TransformerMixin):
    """Takes first derivate of data
    """
    def __init__(self,d=2):
        """Order to use
        Args:
            d (int, optional): derivate order to use. Defaults to 2.
        """
        self.d=d  
        self.fitted_ = False
    def fit(self, X, y=None):
        """not used
        """
        self.fitted_ = True
    def transform(self, X, y=None): 
        """Takes first derivate of data
        Args:
            X (Dataframe): Dataframe of samples [samples x variables] used to train pls_model
            y (_type_, optional): Not used. Defaults to None.
        Returns:
            Dataframe: First derivative of X
        """
        X_=X.diff(self.d,axis=1)
        X_.iloc[:, :self.d] = 0 
        return X_
    def fit_transform(self,X,y=None):
        """Takes first derivate of data
        Args:
            X (Dataframe): Dataframe of samples [samples x variables] used to train pls_model
            y (_type_, optional): Not used. Defaults to None.
        Returns:
            Dataframe: First derivative of X
        """
        return self.transform(X)

#===================================================================================================================================================#

class MSC(BaseEstimator, TransformerMixin):
    """Performs multiplicative scatter correction
    """
    def __init__(self):
        """not used
        """
        self.mean_spectrum_ = None
        self.fitted_ = False
    def fit(self, X, y=None):
        """Takes first derivate of data
        Args:
            X (Dataframe): Dataframe of samples [samples x variables] used to train pls_model
            y (_type_, optional): Not used. Defaults to None.
        """
        self.fitted_ = True
        # Store mean spectrum (as numpy array for performance)
        try: X_ = pd.DataFrame(X)
        except: pass
        self.mean_spectrum_ = X_.mean(axis=0).values
        return self
    def transform(self, X, y=None):
        """Apply MSC

        Args:
            X (Dataframe): Dataframe of samples [samples x variables] used to train pls_model
            y (_type_, optional): Not used. Defaults to None.

        Raises:
            RuntimeError: You must fit the transformer before calling transform.

        Returns:
            Dataframe: MSC of X
        """
        if self.mean_spectrum_ is None:
            raise RuntimeError("You must fit the transformer before calling transform.")
        try: X_ = pd.DataFrame(X)
        except: pass
        X_corrected = []
        ref = self.mean_spectrum_
        for i in range(X_.shape[0]):
            sample = X_.iloc[i].values
            # Linear regression: y = a + b * x
            b, a = np.polyfit(ref, sample, 1)
            corrected = (sample - a) / b
            X_corrected.append(corrected)

        # Return as DataFrame with original index and columns
        return pd.DataFrame(X_corrected, index=X_.index, columns=X_.columns)
    def fit_transform(self, X, y=None):
        """First fit, then transform data

        Args:
            X (Dataframe): Dataframe of samples [samples x variables] used to train pls_model
            y (_type_, optional): Not used. Defaults to None.

        Returns:
            Dataframe: MSC of X
        """
        return self.fit(X).transform(X)

#===================================================================================================================================================#

# Gerretzen, J., Szymańska, E., Jansen, J.J., Bart, J., van Manen, H.J., van den Heuvel, E.R. and Buydens, L.M., 2015. Simple and effective way for data preprocessing selection based on design of experiments. Analytical chemistry, 87(24), pp.12096-12103.
def Gerretzen(model,x,xmin,xmax):
  # Using this solver, None must be placed in first of each step/level
  x_best=None
  y_best=np.inf
  y=-np.inf
  activate=np.ones_like(xmin)
  activate[-1]=0
  xmax=xmax+1
  # First try the raw model
  y_raw=model(xmin)
  y_effect_best=np.inf
  # Test the effects
  for i in range(len(activate)-1):
    x_guess1=xmin
    x_guess1[i]=1
    y_effect=model(x_guess1)
    if y_effect<y_raw:
      activate[i]=1
      x_best=x_guess1
      y_effect_best=y_effect #Update recorded effect
    else:
      activate[i]=0  
  # Random Strategy for non-effect activated
  converged=False
  random_count=1
  
  for _ in range(random_count): #repeat random /////
    for i in range(len(activate)-1):
      if activate[i]==0:
        x_guess1=np.copy(activate) #check this later
        x_guess1[i]=np.random.randint(2,xmax[i])
        y_effect=model(x_guess1)
        if y_effect<y_effect_best:
          activate[i]=1
          x_best=x_guess1
          y_effect_best=y_effect #Update recorded effect
  
  # Sequential Search
  converge=False
  flag=False
  y=-np.inf
  xmin=xmin*activate
  x_guess=xmin
  x_guess[-1]=0 # default
  xmax=(xmax-1)*activate
  xmax[-1]=0 # default
  y_best=y_effect_best
  for g in range(len(xmin)):
    for f in range(1,xmax[g]): #raw data has been tested, no need to start from 0
      x_guess=x_best
      x_guess[g]=f
      x_guess[g+1:]=np.zeros_like(x_guess[g+1:])
      y=model(x_guess)
      #print(x_guess,"  ",y)
      if y<y_best:
        y_best=y
        x_best[g]=x_guess[g] #sequential
  print('Search Completed!')
  return x_best, y_best

#===================================================================================================================================================#

def pp_doe(solutions):
    """Takes solutions from DoE preprocessing and calculates main/interaction effects

    Args:
        solutions (Dataframe): Dataframe of pipelines

    Returns:
        _type_: model describing main/interaction effects (use print(model.summary()))
    """
    doe = []
    target = []
    results = solutions.get_results(verbose=False)

    for i, pipeline in enumerate(solutions.get_pipelines()):
        binary_vector = [0 if step is None else 1 for name, step in pipeline.steps][:-1]
        doe.append(binary_vector)
        target.append(results.iloc[i,2])
    target

    # Convert to DataFrame
    num_factors = len(doe[0])
    factor_names = [f'F{i+1}' for i in range(num_factors)]
    df = pd.DataFrame(doe, columns=factor_names)
    df['MAE'] = target

    # Generate 2-way interaction terms
    for combo in combinations(factor_names, 2):
        colname = f"{combo[0]}*{combo[1]}"
        df[colname] = df[combo[0]] * df[combo[1]]

    # Define X and y
    X = sm.add_constant(df.drop(columns='MAE'))  # add intercept
    y = df['MAE']

    # Fit linear regression model
    model = sm.OLS(y, X).fit()

    # Print model summary
    print(model.summary())
    
    return model