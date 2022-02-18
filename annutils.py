from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
#
import tensorflow as tf
from tensorflow import keras
import joblib
#
import pandas as pd
import numpy as np
# viz
import hvplot.pandas
import holoviews as hv
import panel as pn

def synchronize(dfin, dfout):
    '''
    synchronizes on index dfx and dfy and return tuple of synchronized data frames
    '''
    dfsync = pd.concat([dfin,dfout],axis=1).dropna()
    return dfsync.iloc[:,:len(dfin.columns)], dfsync.iloc[:,len(dfin.columns):]

def create_antecedent_inputs(df,ndays=8,window_size=11,nwindows=10):
    '''
    create data frame for CALSIM ANN input
    Each column of the input dataframe is appended by :-
    * input from each day going back to 7 days (current day + 7 days) = 8 new columns for each input
    * 11 day average input for 10 non-overlapping 11 day periods, starting from the 8th day = 10 new columns for each input

    Returns
    -------
    A dataframe with input columns = (8 daily shifted and 10 average shifted) for each input column

    '''
    arr1=[df.shift(n) for n in range(ndays)]
    dfr=df.rolling(str(window_size)+'D',min_periods=window_size).mean()
    arr2=[dfr.shift(periods=(window_size*n+ndays),freq='D') for n in range(nwindows)]
    df_x=pd.concat(arr1+arr2,axis=1).dropna()# nsamples, nfeatures
    return df_x

def trim_output_to_index(df,index):
    '''
    helper method to create output of a certain size ( typically to match the input )
    '''
    return df.loc[index,:] #nsamples, noutput

def split(df, calib_slice, valid_slice):
    return df[calib_slice], df[valid_slice]

def create_xyscaler(dfin,dfout):
    xscaler=MinMaxScaler()
    xx=xscaler.fit_transform(dfin)
    #
    yscaler=MinMaxScaler()
    yy=yscaler.fit_transform(dfout)
    return xscaler, yscaler

def _old_create_xyscaler(dfin,dfout):
    return create_xyscaler(pd.concat(dfin,axis=0), pd.concat(dfout,axis=0))

def create_training_sets(dfin, dfout, calib_slice=slice('1940','2015'), valid_slice=slice('1923','1939')):
    '''
    dfin is a dataframe that has sample (rows/timesteps) x nfeatures 
    dfout is a dataframe that has sample (rows/timesteps) x 1 label
    Both these data frames are assumed to be indexed by time with daily timestep

    This calls create_antecedent_inputs to create the CALSIM 3 way of creating antecedent information for each of the features

    Returns a tuple of two pairs (tuples) of calibration and validation training set where each set consists of input and output
    it also returns the xscaler and yscaler in addition to the two tuples above
    '''
    # create antecedent inputs aligned with outputs for each pair of dfin and dfout
    dfina,dfouta=[],[]
    # scale across all inputs and outputs
    xscaler,yscaler=_old_create_xyscaler(dfin,dfout)
    for dfi,dfo in zip(dfin,dfout):
        dfi,dfo=synchronize(dfi,dfo)
        dfi,dfo=pd.DataFrame(xscaler.transform(dfi),dfi.index,columns=dfi.columns),pd.DataFrame(yscaler.transform(dfo),dfo.index,columns=dfo.columns)
        dfi,dfo=synchronize(create_antecedent_inputs(dfi),dfo)
        dfina.append(dfi)
        dfouta.append(dfo)
    # split in calibration and validation slices
    dfins=[split(dfx,calib_slice,valid_slice) for dfx in dfina]
    dfouts=[split(dfy,calib_slice,valid_slice) for dfy in dfouta]
    # append all calibration and validation slices across all input/output sets
    xallc,xallv=dfins[0]
    for xc,xv in dfins[1:]:
        xallc=np.append(xallc,xc,axis=0)
        xallv=np.append(xallv,xv,axis=0)
    yallc, yallv = dfouts[0]
    for yc,yv in dfouts[1:]:
        yallc=np.append(yallc,yc,axis=0)
        yallv=np.append(yallv,yv,axis=0)
    return (xallc,yallc),(xallv,yallv),xscaler,yscaler

def create_memory_sequence_set(xx,yy,time_memory=120):
    '''
    given an np.array of xx (features/inputs) and yy (labels/outputs) and a time memory of steps
    shape[0] of the array represents the steps (usually evenly spaced time)
    return a tuple of inputs/outputs sampled for every step going back to time memory
    The shape of the returned arrays is dictated by keras 
    inputs.shape (nsamples x time_memory steps x nfeatures)
    outputs.shape (nsamples x nlabels)
    '''
    xxarr=[xx[i:time_memory+i,:] for i in range(xx.shape[0]-time_memory)]
    xxarr=np.expand_dims(xxarr,axis=0)[0]
    yyarr=[yy[time_memory+i] for i in range(xx.shape[0]-time_memory)]
    yyarr=np.array(yyarr)[:,0]
    return xxarr,yyarr
 
############### TESTING - SPLIT HERE #####################
import panel as pn

def predict(model,dfx,xscaler,yscaler):
    dfx=pd.DataFrame(xscaler.transform(dfx),dfx.index,columns=dfx.columns)
    xx=create_antecedent_inputs(dfx)
    oindex=xx.index
    yyp=model.predict(xx)
    dfp=pd.DataFrame(yscaler.inverse_transform(yyp),index=oindex,columns=['prediction'])
    return dfp

def predict_with_actual(model, dfx, dfy, xscaler, yscaler):
    dfp=predict(model, dfx, xscaler, yscaler)
    return pd.concat([dfy,dfp],axis=1).dropna()
    
def plot(dfy,dfp):
    return dfy.hvplot(label='target')*dfp.hvplot(label='prediction')

def show_performance(model, dfx, dfy, xscaler, yscaler):
    from sklearn.metrics import r2_score
    dfyp=predict_with_actual(model,dfx,dfy,xscaler,yscaler)
    print('R^2 ',r2_score(dfyp.iloc[:,0],dfyp.iloc[:,1]))
    dfyp.columns=['target','prediction']
    plt=(dfyp.iloc[:,1]-dfyp.iloc[:,0]).hvplot.kde().opts(width=300)+dfyp.hvplot.points(x='target',y='prediction').opts(width=300)
    return pn.Column(plt, plot(dfyp.iloc[:,0],dfyp.iloc[:,1]))
###########
import joblib
class ANNModel:
    '''
    model consists of the model file + the scaling of inputs and outputs
    '''
    def __init__(self,model_name, model,xscaler,yscaler):
        self.model_name = model_name
        self.model=model
        self.xscaler=xscaler
        self.yscaler=yscaler
    def predict(self, dfin):
        return predict(self.model,dfin,self.xscaler,self.yscaler)
#
def save_model(model_name, model, xscaler, yscaler):
    '''
    save keras model and scaling to files
    '''
    joblib.dump((xscaler,yscaler),'%s-xyscaler.dump'%model_name)
    model.save('%s.h5'%model_name)

def load_model(model_name):
    '''
    load model (ANNModel) which consists of model (Keras) and scalers loaded from two files
    '''
    model=keras.models.load_model('%s.h5'%model_name)
    xscaler,yscaler=joblib.load('%s-xyscaler.dump'%model_name)
    return ANNModel(model_name, model,xscaler,yscaler)

########### TRAINING - SPLIT THIS MODULE HERE ###################

def train_nn(x,y,hidden_layer_sizes=(10,),max_iter=1000,activation='relu',tol=1e-4):
    mlp=MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,max_iter=max_iter,activation=activation, tol=tol)
    mlp.fit(x,y)
    return mlp

def _old_predict(df_x, mlp, xs, ys):
    y_pred=mlp.predict(xs.transform(df_x))
    y_pred=ys.inverse_transform(np.vstack(y_pred))
    return pd.DataFrame(y_pred,df_x.index,columns=['prediction'])

def show(df_x, df_y, mlp, xs, ys):
    y=np.ravel(ys.transform(df_y))
    y_pred=mlp.predict(xs.transform(df_x))
    r2=mlp.score(xs.transform(df_x),y)
    print('Score: ',r2)
    return pn.Column(pn.Row(hv.Scatter((y,y_pred)).opts(aspect='square'),hv.Distribution(y_pred-y).opts(aspect='square')),
                     hv.Curve((df_y.index,y_pred),label='prediction')*hv.Curve((df_y.index,y),label='target').opts(width=800))

def train(df_x,df_y,hidden_layer_sizes=(10,),max_iter=1000,activation='relu',tol=1e-4):
    xs=MinMaxScaler()
    x=xs.fit_transform(df_x)
    ys=MinMaxScaler()
    y=np.ravel(ys.fit_transform(df_y))
    mlp=train_nn(x, y,hidden_layer_sizes=hidden_layer_sizes,max_iter=max_iter,activation=activation,tol=tol)
    return mlp, xs, ys

def train_more(mlp,xs,ys,df_x,df_y):
    x=xs.transform(df_x)
    y=np.ravel(ys.transform(df_y))
    mlp.fit(x,y)
    return mlp

