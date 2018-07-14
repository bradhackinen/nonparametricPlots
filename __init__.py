import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skmisc import loess
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor


def loessPlot(X,y,scatter=True,res=100,x_min=None,x_max=None,x_plot=None,ci_alpha=0.05,scatter_kws={},line_kws={},fill_kws={},**loess_args,):
    '''
    Plots a loess curve with shaded confidence confidence intervals using
    scikit-misc loess function (https://has2k1.github.io/scikit-misc/loess.html)
    -x can be a (n,) or (n,k) ndarray. If x is (n,k), the x-axis of the plot will
    correspond to the first covariate in the first column, with the other covariates
    entering as invisible controls.
    -y must be a (n,) ndarray
    -res,x_min,x_max set the resolution and domain for sampling the loess prediction.
    If x_min and x_max are not provided they are set to the min and max of the first
    dimension of x.
    -x_plot (optional) overrides res,x_min,x_max and sets the sampling points for
    the plot directly. Must be a 1-d ndarray.
    -ci_alpha sets confidence interval alpha parameter (default=0.05)
    -Additional loess args can be passed as named parameters
    '''
    #Set default arguments for graphic elements
    scatter_args = {'s':10,'linewidth':0}
    scatter_args.update(scatter_kws)

    line_args = {}
    line_args.update(line_kws)

    fill_args = {'color':'k','alpha':0.25,'linewidth':0}
    fill_args.update(fill_kws)

    #Split off first dimension of X for plotting
    if len(X.shape) > 1:
        x = X[:,0]
    else:
        x = X
        X = X[:,np.newaxis]

    #Sort out plot range and sampling points
    if x_min is None:
        x_min = x.min()

    if x_max is None:
        x_max = x.max()

    if x_plot is None:
        x_plot = np.linspace(x_min,x_max,res)

    #Compute loess curve and confidence intervals
    loessObject = loess.loess(X,y,**loess_args)
    prediction = loessObject.predict(x_plot,True)
    confidence_intervals = prediction.confidence(alpha=ci_alpha)

    if ax is None:
        ax = plt.gca()

    #Plot
    if scatter:
        ax.scatter(x,y,**scatter_args)

    ax.plot(x_plot,prediction.values,**line_args)
    ax.fill_between(x_plot,confidence_intervals.upper,confidence_intervals.lower,**fill_args)

    return ax



def kNeighbours2dPlot(X,y,k=10,res=100,dist_scale='normalize',im_kws={},reg_kws={},ax=None):
    if isinstance(X,pd.core.frame.DataFrame):
        X = X.values

    if 'origin' not in reg_kws:
        im_kws['origin'] ='lower'

    if 'extent' not in im_kws:
        im_kws['extent'] = (X[:,0].min(),X[:,0].max(),X[:,1].min(),X[:,1].max())

    if  'aspect' not in im_kws:
        im_kws['aspect'] = (X[:,0].max()-X[:,0].min())/(X[:,1].max()-X[:,1].min())

    if dist_scale is not None:
        if dist_scale == 'normalize':
            X = X/(X.max(axis=0) - X.min(axis=0))
        else:
            X = X/dist_scale

    kneighbours = KNeighborsRegressor(n_neighbors=k,**reg_kws)
    kneighbours.fit(X,y)

    xx,yy = np.meshgrid(np.linspace(X[:,0].min(),X[:,0].max(),res),np.linspace(X[:,1].min(),X[:,1].max(),res))
    X_grid = np.vstack([xx.ravel(),yy.ravel()]).T

    y_hat = kneighbours.predict(X_grid)
    Y_hat = y_hat.reshape((res,res))
    if ax is None:
        return plt.imshow(Y_hat,**im_kws)
    else:
        return ax.imshow(Y_hat,**im_kws)


def rNeighbours2dPlot(X,y,r=0.5,res=100,dist_scale='normalize',im_kws={},reg_kws={},ax=None):
    if isinstance(X,pd.core.frame.DataFrame):
        X = X.values

    if 'origin' not in reg_kws:
        im_kws['origin'] ='lower'

    if 'extent' not in im_kws:
        im_kws['extent'] = (X[:,0].min(),X[:,0].max(),X[:,1].min(),X[:,1].max())

    if  'aspect' not in im_kws:
        im_kws['aspect'] = (X[:,0].max()-X[:,0].min())/(X[:,1].max()-X[:,1].min())

    if dist_scale is not None:
        if dist_scale == 'normalize':
            X = X/(X.max(axis=0) - X.min(axis=0))
        else:
            X = X/dist_scale

    kneighbours = RadiusNeighborsRegressor(radius=r,**reg_kws)
    kneighbours.fit(X,y)

    xx,yy = np.meshgrid(np.linspace(X[:,0].min(),X[:,0].max(),res),np.linspace(X[:,1].min(),X[:,1].max(),res))
    X_grid = np.vstack([xx.ravel(),yy.ravel()]).T

    y_hat = kneighbours.predict(X_grid)
    Y_hat = y_hat.reshape((res,res))
    if ax is None:
        return plt.imshow(Y_hat,**im_kws)
    else:
        return ax.imshow(Y_hat,**im_kws)





def localPartialCorr(y1,y2,X,res=100,x_min=None,x_max=None,x_plot=None,ci_alpha=0.05,inner_bw=10,ax=None,line_kws={},fill_kws={},**loess_args):
    '''
    Computes the local partial correlation betwen y1 and y2 given controls X

    WARNING: Standard errors should not be taking seriously. Need to update code to
    estimate them more precisely, or at least correct for intermediate smoothing step.
    '''
    #Split off first dimension of X for defining distances
    if len(X.shape) > 1:
        x = X[:,0]
    else:
        x = X
        X = X[:,np.newaxis]

    #Sort out plot range and sampling points
    if x_min is None:
        x_min = x.min()

    if x_max is None:
        x_max = x.max()

    if x_plot is None:
        x_plot = np.linspace(x_min,x_max,res)

    #Compute local residuals
    loessObjects = [loess.loess(X,y,**loess_args) for y in (y1,y2)]

    for loessObject in loessObjects:
        loessObject.fit()

    r1,r2 = [loessObject.outputs.fitted_residuals for loessObject in loessObjects]

    #Compute invariant components of weighted correlation
    r11 = r1**2
    r22 = r2**2
    r12 = r1*r2

    #Compute local kernal bandwidth for each plot point
    n_span = int(inner_bw)
    h = np.zeros(res)
    for i in range(res):
        d = np.abs(x - x_plot[i])
        h[i] = np.partition(d,n_span)[n_span]

    #Compute locally weighted correlation for each x_plot point
    corr = np.zeros(res)
    for i in range(res):
        #Construct weight matrix using tri-cube weight function
        d = np.abs(x - x_plot[i])
        w = (1 - (d/h[i])**3)**3
        w = np.clip(w,a_min=0,a_max=None)

        #Compute weighted correlation
        corr[i] = np.dot(w,r12) / np.sqrt(np.dot(w,r11)*np.dot(w,r22))

    return loessPlot(x_plot,corr,x_plot=x_plot,ax=ax,scatter=False,line_kws=line_kws,fill_kws=fill_kws,**loess_args)





# # Test code for local loess -------------------------------------------------
# #Build some test data
# testDF = pd.DataFrame(index=list(range(10000)))
# testDF['x'] = np.linspace(0,1,num=len(testDF))
# testDF['rho'] = np.sin(testDF['x']*6.28)
# testDF['y1'] = np.random.normal(size=len(testDF))
# testDF['y2'] = testDF['rho']*testDF['y1'] + np.sqrt(1 - testDF['rho']**2)*np.random.normal(size=len(testDF))
#
# testDF['bin'] = (testDF['x']*20).astype(int)
#
# testDF.plot('x','rho',c='k')
#
# bin_corr = np.zeros(testDF['bin'].nunique())
# bin_x = np.zeros(testDF['bin'].nunique())
# for i,df in testDF.groupby('bin'):
#     bin_corr[i] = df[['y1','y2']].corr().values[0,1]
#     bin_x[i] = df['x'].mean()
#
# plt.plot(bin_x,bin_corr)
#
# localLoessCorr(testDF['y1'],testDF['y2'],testDF['x'],degree=2,inner_bw=10,span=0.5)
