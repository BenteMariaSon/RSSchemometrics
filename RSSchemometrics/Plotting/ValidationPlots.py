import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score, root_mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
# from Codebase import rucolors
import Plotting.rucolors as rucolors

#===================================================================================================================================================#

def predvsref_plot(y_pred, y, xlabel=None, ylabel=None, title=None, color=rucolors.red, ax_in=None):
    """Plots predicted vs reference values

    Parameters:
    - y_pred (ndarray): array containing the predicted values
    - y (ndarray): array containing the reference/actual values
    - xlabel (string, optional): Description of x axis. Defaults to None.
    - ylabel (string, optional): Description of y axis. Defaults to None.
    - title (string, optional): Description of title. Defaults to None.
    - color (color, optional): the color to give to the plot (can be rgb values or string conaining hex code). Defaults to rucolors.red (radboud red)
    - ax_in (figure axis, optional): Axis can be passed here to add the scatter plot to an existing axis system. Defaults to None
    
    Output:
    - Displays the predicted vs reference plot
    """
    #Metrics
    R2 = r2_score(y, y_pred)
    RMSE = root_mean_squared_error(y, y_pred)
    minimum = np.min([np.min(y_pred), np.min(y)])
    maximum = np.max([np.max(y_pred), np.max(y)])
    #Plot
    if ax_in is None:
        fig, ax = plt.subplots(figsize=(6,4))
    else:
        ax = ax_in
        fig = ax_in.figure

    ax.scatter(y, y_pred, color=color)
    ax.plot(np.linspace(minimum, maximum), np.linspace(minimum, maximum), c='#797777', linestyle="--")
    ax.set_xlabel(xlabel if xlabel is not None else "True Reference Values")
    ax.set_ylabel(ylabel if ylabel is not None else "Predicted Values")
    ax.set_title(title if title is not None else "Predicted vs Reference Values")
    ax.text(0.05, 0.90, f'R2: {np.round(R2,3)}\nRMSE: {np.round(RMSE,3)}', transform=ax.transAxes)
    if ax_in is None:
        fig.tight_layout()
        plt.show()
    else:
        return fig, ax
    
#===================================================================================================================================================#

def residuals_plot(y_pred, y, xlabel=None, ylabel=None, title=None, color=rucolors.red, ax_in=None):
    """Creates a residuals plot with a y=0 reference line.

    Parameters:
    - y_pred (ndarray): array containing the predicted values
    - y (ndarray): array containing the reference/actual values
    - xlabel (string, optional): Description of x axis. Defaults to None.
    - ylabel (string, optional): Description of y axis. Defaults to None.
    - title (string, optional): Description of title. Defaults to None.
    - color (color, optional): the color to give to the plot (can be rgb values or string conaining hex code). Defaults to rucolors.red (radboud red)
    - ax_in (figure axis, optional): Axis can be passed here to add the scatter plot to an existing axis system. Defaults to None
    
    Output:
    - Displays the residuals plot
    """
    residuals = y_pred - y
    if ax_in is None:
        fig, ax = plt.subplots(figsize=(6,4))
    else:
        ax = ax_in
        fig = ax_in.figure
    ax.scatter(y_pred, residuals, color=color)
    ax.axhline(0, linestyle='--', color='#797777')
    ax.set_xlabel(xlabel if xlabel is not None else "Predicted values")
    ax.set_ylabel(ylabel if ylabel is not None else "Residuals")
    ax.set_title(title if title is not None else "Residuals vs. Predicted")
    
    if ax_in is None:
        fig.tight_layout()
        plt.show()
    else:
        return fig, ax
    
#===================================================================================================================================================#

def coefficients_plot(coefficients, x_features, xlabel=None, ylabel=None, title=None, color=rucolors.red, ax_in=None):
    """Creates a coefficients plot, showing the coefficients values as a bar for each feature.

    Parameters:
    - coefficients (Dataframe): dataframe containing the coefficients
    - x_features (str-array): names of the features to be plotted
    - xlabel (string, optional): Description of x axis. Defaults to None.
    - ylabel (string, optional): Description of y axis. Defaults to None.
    - title (string, optional): Description of title. Defaults to None.
    - color (color, optional): the color to give to the plot (can be rgb values or string conaining hex code). Defaults to rucolors.red (radboud red)
    - ax_in (figure axis, optional): Axis can be passed here to add the scatter plot to an existing axis system. Defaults to None
        
    Output: 
    - Displays the coefficients plot
    """
    coeffs = pd.Series(coefficients, index=x_features)
    if ax_in is None:
        fig, ax = plt.subplots(figsize=(6,4))
    else:
        ax = ax_in
        fig = ax_in.figure

    coeffs.plot(kind='barh', color=color, edgecolor='black')
    ax.set_title(title if title else "Regression Coefficients (B)")
    ax.set_xlabel(xlabel if xlabel else "Coefficient Value")
    ax.set_ylabel(ylabel if ylabel else "Variables")
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    if ax_in is None:
        fig.tight_layout()
        plt.show()
    else:
        return fig, ax

#===================================================================================================================================================#

def confusion_matrix_plot(y_pred, y, labels=None, title=None, cmap=rucolors.RU_colormap, ax_in=None):
    """Dispaly the confusion matrix as a plot

    Parameters:
    - y_pred (ndarray): array containing the predicted values
    - y (ndarray): array containing the reference/actual values
    - labels (str-array): array containing the names of each of the classes, defaults to None
    - title (string, optional): Description of title. Defaults to None.
    - cmap (cmap or str, optional): A colormap or string containing the name of the colormap that should be used to color the points. Defaults to rucolors.RU_colormap (this is a custom colormap)
    - ax_in (figure axis, optional): Axis can be passed here to add the scatter plot to an existing axis system. Defaults to None
    
    Output:
    - Displays the confusion matrix
    """
    if ax_in is None:
        fig, ax = plt.subplots(figsize=(6,4))
    else:
        ax = ax_in
        fig = ax_in.figure

    cm = confusion_matrix(y, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=cmap, values_format='d', ax=ax)
    ax.set_title(title if title else "Confusion Matrix")
    ax.grid(False)

    if ax_in is None:
        fig.tight_layout()
        plt.show()
    else:
        return fig, ax