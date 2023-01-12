# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 21:30:03 2023

@author: mahdi
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime,date,timedelta
from pyod.models.knn import KNN
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import CubicSpline

##########################################################   
def plotter(dataset,x,y=None,**kwargs):
    # this function plot all diagrams which we need in this paper, including time series, histogram and scatter plot where kwargs 
    # argument includes label of x and y axes, color, label, xlim, ylim, number of bins in histogram and point size in scatter plot. 
    
    kind=input('please write which type of plot do you need among one of (1) time series, (2) line (3) hist and (4) scatter: ')
    
    if kind=='time series':
        plt.plot(dataset.index,dataset[x],label=kwargs.get('label', None),color=kwargs.get('color', None),marker=kwargs.get('marker', None))
    elif kind=='line':       
        plt.plot(dataset[x],dataset[y],label=kwargs.get('label', None),color=kwargs.get('color', None),marker=kwargs.get('marker', None))
    elif kind=='hist':
        plt.hist(dataset[x],label=kwargs.get('label', None),color=kwargs.get('color', None),bins=kwargs.get('bins', None))
    elif kind=='scatter':
        plt.scatter(dataset[x],dataset[y],label=kwargs.get('label', None),color=kwargs.get('color', None),s=kwargs.get('size', None),marker=kwargs.get('marker', None))    
    plt.xlabel(kwargs.get('xlabel', None))
    plt.ylabel(kwargs.get('ylabel', None))
    plt.xlim(kwargs.get('xlim', None))
    plt.ylim(kwargs.get('ylim', None))
    plt.legend()
    return

##########################################################
def binned_average(dataset,x,y,division_number):
    # this function bin x values and calculate the average of y values for each bin. Then return the middle of bins vs average of 
    # y for each bin
    interval=(dataset[x].max()-dataset[x].min())/division_number
    
    bins=np.arange(dataset[x].min(),dataset[x].max(),interval)
    mid_bins=(bins[1:]+bins[:-1])/2.0
    mid_bins=pd.DataFrame(mid_bins,columns=[x])
    
    median_y=pd.DataFrame(dataset[y].groupby(pd.cut(dataset[x],bins)).median(),columns=[y])
    
    std_y=pd.DataFrame(dataset[y].groupby(pd.cut(dataset[x],bins)).std())
    
    median_y.index=mid_bins.index
    std_y.index=mid_bins.index
    binned_mean=pd.concat([mid_bins,median_y,std_y],axis=1)
    binned_mean.columns=[x,y,'std_'+y]
    return binned_mean

##############################     
def moving_mean_std(dataset):
    # To perform the first step of QA methodology introduced in the paper, this function calculates moving average and 
    # moving standard deviation for the input dataset and return two dataframes as moving_mean and moving_std. 
    
    moving_mean=dataset.groupby([dataset.index.year,dataset.index.month]).mean()
    moving_std=dataset.groupby([dataset.index.year,dataset.index.month]).std()

    index_list=[]
    for index in moving_mean.index:
        x=datetime.strptime(format(datetime(index[0],index[1],15),'%d%m/%Y'),'%d%m/%Y').date()
        index_list.append(x)
    
    moving_mean.index=index_list
    moving_std.index=index_list
    
    return moving_mean,moving_std

##########################################################
def turbine_wake_angle_detector(dataset,phi,PHI_PRIME_0,PHI_PRIME_1,DELTA_PHI,inverse_manu_power_curve,rated_u,u,p,interval,power_min=0,power_max=1950):
    # This function plots the deviation between real wind speed and manufacture wind speed for different angles. Later we use
    # the plot to see for which wind direction wind speed, measured at the mast, are affected by interference of neighboring wind turbine.  
    
    power_bins=np.arange(power_min,power_max,interval)
    power_center=(power_bins[1:]+power_bins[:-1])/2.0
    power_center=pd.DataFrame(power_center,columns=[p])
    
    manu_u=pd.DataFrame(inverse_manu_power_curve(power_center),columns=['standard '+u])
    
    u_deviation_dic={'wind direction':[],'ws deviation':[]}
    
    for wd in np.arange(PHI_PRIME_0,PHI_PRIME_1,DELTA_PHI):
        real_u=pd.DataFrame(dataset.loc[(dataset[phi]>wd)&(dataset[phi]<wd+DELTA_PHI)&(dataset[u]<rated_u),u].groupby(pd.cut(dataset.loc[(dataset[phi]>wd)&(dataset[phi]<wd+DELTA_PHI)&(dataset[u]<rated_u),p],power_bins)).median(),columns=[u])
        real_u.index=power_center.index
    
        df=pd.concat([manu_u,real_u,power_center],axis=1)
        
        u_deviation=(df['standard '+u]-df[u]).mean()
        
        u_deviation_dic['wind direction'].append(wd)
        u_deviation_dic['ws deviation'].append(u_deviation)
        
    u_deviation_dic=pd.DataFrame(u_deviation_dic)
        
    return u_deviation_dic,df
   
##########################################################
def icing(dataset,t,t_c,u_c,sonic_u,cup_u,cup_u_prime):
    # this function finds the values of wind speed affected by icing and replace them by NAN, using the algorithm in the paper. 
    # t and t_c is temperature and critical temperature and u_c is critical value for difference in wind speed. sonic_u is sonic wind speed
    # and cup_u and cup_u_prime are wind speeds measured by cup anemometer respectively at the same and other height with sonic anemometer. 
    icing_times=pd.DatetimeIndex([])
    for u in cup_u:
        dates=dataset[(dataset[t]<t_c)&(dataset[sonic_u]-dataset[u]>u_c)].index
        icing_times=icing_times.union(dates)
    
    cup_u=cup_u+cup_u_prime
    dataset.loc[icing_times,cup_u]=np.nan
    return dataset

##########################################################
def fixed_limit(dataset,limits):
    # This function set a fixed limit which is constant over the whole period of measurement. 
    for i in range(len(limits)):
        for j in range(len(list(limits.keys())[i])):
            dataset[list(limits.keys())[i][j]]=dataset[list(limits.keys())[i][j]].where(dataset[list(limits.keys())[i][j]].between(list(limits.values())[i][0],list(limits.values())[i][1]))   
    return dataset

##########################################################
def block_identifier(dataset,target_var):
    # this function gets the dataset and the target variable and returns both block length (the number of consecutive duplicates within 
    # each block) and block index for which we have that block length. We use the output of this function to plot the histogram later.  
    empty_array=np.empty((0,1))
    
    x=dataset[target_var].ne(dataset[target_var].shift()).cumsum()
    y=dataset[target_var].groupby(x).transform('count')
    y=pd.DataFrame(y)
    y=pd.DataFrame(y[y.columns[0]].value_counts())
    y=pd.DataFrame(y[y.columns[0]]/y.index)
    
    y.reset_index(inplace=True)
    y.columns=['block length','block number']
    y=y[y['block length']>1.0]
    
    for i in range(y.shape[0]):
        empty_array=np.append(empty_array,y.iloc[i]['block length']*np.ones(int(y.iloc[i]['block number'])))
    
    empty_array=pd.DataFrame(empty_array,columns=['block length'])
    empty_array.index.name='block index'

    return empty_array

##########################################################
def duplicates_date_identifier(dataset,target_var,dead_number):
    # this function gets the dataset, target variable and the critical values of block length and returns suspicious date for 
    # those duplicates.  
    
    y=dataset[target_var].ne(dataset[target_var].shift()).cumsum()
    suspicious_duplicate_date=dataset[target_var][dataset[target_var].groupby(y).transform('count')>dead_number].index
        
    return suspicious_duplicate_date

##########################################################
def repeated_constant_remover(dataset,critical_block_lengths):
    # this function replaces the values in the blocks, with block length>critical block length, with NaN. 
    for key in critical_block_lengths.keys():
        target_var=key
        critical_block_length=critical_block_lengths[key]
    
        duplicates_dates=duplicates_date_identifier(dataset,target_var,critical_block_length)
        dataset.loc[duplicates_dates,key]=np.nan
    
    return dataset

##########################################################
def confidence_interval(dataset,var_1,var_2,division_number,lower_percentage,higher_percentage):
    # this function divide the range of var_1 and for each division it calculates percentile for var_2 distribution to identify the 
    # data out of this interval (confidence interval) as suspicious data, maybe outliers.
    
    x_set=np.linspace(dataset[var_1].min(),dataset[var_1].max(),division_number)
    means=np.array([(x_set[i]+x_set[i+1])/2.0 for i in range(len(x_set)-1)])

    percentiles=np.empty((0,3))
    suspicous_date=pd.DataFrame({})

    for i in range(len(x_set)-1):
        a=dataset[(dataset[var_1]>=x_set[i])&((dataset[var_1]<x_set[i+1]))]

        lower_percentile=np.nanpercentile(a[var_2],lower_percentage)   
        higher_percentile=np.nanpercentile(a[var_2],higher_percentage) 
        median=np.nanpercentile(a[var_2],50.0)
    
        percentiles=np.append(percentiles,np.array([[median,lower_percentile,higher_percentile]]),axis=0)
   
        suspicous_date=pd.concat([suspicous_date,pd.DataFrame(a[(a[var_2]<lower_percentile)|(a[var_2]>higher_percentile)].index)],axis=0)
      
    percentiles=pd.DataFrame(percentiles,columns=['median','lower percentile','higher percentile'])
    percentiles[var_1]=means
    
    suspicous_date.set_index('time',inplace=True)
    suspicous_date=suspicous_date.index
           
    return percentiles,suspicous_date

##########################################################
def inconsistency_remover(dataset,conf_dic,division_number):
    # this function replaces inconsistent values, existing out of confidence interval, with NaN. 
    for key in conf_dic.keys():
        target_var=key
        counterpart_var=conf_dic[key][0]
        percentiles,suspicous_date=confidence_interval(dataset,target_var,counterpart_var,division_number,conf_dic[key][1],conf_dic[key][2])
        dataset.loc[suspicous_date,[target_var,counterpart_var]]=np.nan
    return dataset

##########################################################
def extreme_variation_identifier(dataset,extreme_dic):
    # this function gets the dataset and extreme_dic which includes target variable and critical number and replaces the values 
    # for which the difference between two consecutive steps are higher than critical number. 
    for key in extreme_dic.keys():
        var=key     
        variation=pd.DataFrame(np.abs(dataset[var].diff()))  
        extreme_date=variation[variation[var]>extreme_dic[key]].index   
        dataset.loc[extreme_date,key]=np.nan
    return dataset

##########################################################
def daily_gust(dataset,var,not_null_number):
    # this function gets the dataset and takes the day for which we have, at least, "not_null_percentage" percent of not null values. 
    # Then calculates ln(gust) and ln(daily mean wind speed) as a dataframe. 
    
    x1=dataset[var].groupby([dataset[var].index.year,dataset[var].index.month,dataset[var].index.day]).count()
    x1=x1[x1>not_null_number]
    x1=pd.DataFrame(x1.index.values.tolist(),columns=['year','month','day'])
    x1=pd.to_datetime(x1)

    dates=pd.date_range(x1[0],x1[0]+timedelta(days=1),freq='10min')[:-1]
    for t in range(1,len(x1)):
        added_date=pd.date_range(x1[t],x1[t]+timedelta(days=1),freq='10min')[:-1]
        dates=dates.append(added_date) 

    u_daily_max=dataset.loc[dates,var].groupby([dataset.loc[dates,var].index.year,dataset.loc[dates,var].index.month,dataset.loc[dates,var].index.day]).max()
    u_daily_mean=dataset.loc[dates,var].groupby([dataset.loc[dates,var].index.year,dataset.loc[dates,var].index.month,dataset.loc[dates,var].index.day]).mean()
    G=(u_daily_max/u_daily_mean)-1.0

    ln_G=pd.DataFrame(np.log(G))
    ln_u_daily_mean=pd.DataFrame(np.log(u_daily_mean))
    gust_df=pd.concat([ln_u_daily_mean,ln_G],axis=1)
    gust_df.columns=['ln u','ln g']
    gust_index=pd.DataFrame(gust_df.index.values.tolist(),columns=['year','month','day'])
    gust_index=pd.to_datetime(gust_index)
    gust_df.index=gust_index
    
    return gust_df

###############################################################
def outlier_detector(dataset,model,outliers_fraction,minimum_anomaly_score):
    ## This function takes uncleaned dataset and return cleaned one after detction and removal of outliers. 
    column_list=dataset.columns
    dataset_index=pd.DataFrame(dataset.index)
    dataset_index.columns=['date time']
    dataset_index.set_index('date time',inplace=True)
    
    scaler=StandardScaler()
    dataset=scaler.fit_transform(dataset)
    
    model.fit(dataset)
    outlier_predictor=model.predict(dataset)
    anomaly_score=model.decision_function(dataset)
    
    outlier_index=np.where(outlier_predictor==1)[0]
    inlier_index=np.where(outlier_predictor==0)[0]
    
    for j in outlier_index:
        if anomaly_score[j]<minimum_anomaly_score:
            inlier_index=np.append(inlier_index,[j])
            outlier_index=np.delete(outlier_index,np.argwhere(outlier_index==j))     
    
    outliers=dataset[outlier_index]
    inliers=dataset[inlier_index]
    
    outliers=scaler.inverse_transform(outliers)
    inliers=scaler.inverse_transform(inliers)
    
    outlier_index=dataset_index.iloc[outlier_index]
    inlier_index=dataset_index.iloc[inlier_index]
    
    outliers=pd.DataFrame(outliers)
    outliers.columns=column_list
    outliers.index=outlier_index.index
    inliers=pd.DataFrame(inliers)
    inliers.columns=column_list
    inliers.index=inlier_index.index
    
    return inliers,outliers

##########################################################
def mast_interference(dataset,tower_wake_dic,phi):
    # this function removes the values measured by the sensor in wake region of the tower with NaN   
    for key in tower_wake_dic.keys():
        u=key
        phi_0=tower_wake_dic[key][0]
        phi_1=tower_wake_dic[key][1] 
        dataset.loc[(dataset[phi]>=phi_0)&(dataset[phi]<=phi_1),u]=np.nan
    
    return dataset

##########################################################
def transfromer(dataset):
    # this function transfrom the input dataset and return transfromed dataset,using the equation 4 in the paper 
    inv_df=pd.DataFrame(columns=dataset.columns)
    for var in dataset.columns:
        inv_df[var]=1.0/((dataset[var]-dataset[var].min())/(dataset[var].max()-dataset[var].min()))
    return inv_df

##########################################################
def small_outlier_remover(dataset,low_dic,lower_percentage,higher_percentage,division_number):
    # if the target variable is (1) very small and (2) internally inconsistent, this function replaces with NaN 
    for key in low_dic.keys():
        var_1=key
        var_2=low_dic[key][0]
        low_df=dataset[dataset[key]<((dataset[key].max()-dataset[key].min())/(low_dic[key][1]))+dataset[key].min()]
        percentiles,dates=confidence_interval(low_df,var_1,var_2,division_number,lower_percentage,higher_percentage)
        dataset.loc[dates,[var_1,var_2]]=np.nan
    
    return dataset

##########################################################
def pressure_extrapolator(dataset,z_0,pressure,temperature,r_d=287.0,g=9.8,gamma=-0.006):
    # This function extrapolate pressure and calculate its value at 75m using the value of pressure at 1m. the assumption is 
    # that the lapse rate is constant for different heights. In addition to constant lapse rate assumption, we can assume other 
    # scenarios for temperature profile to extrapolate the pressure, using this function. we just have extend. 
    dataset[pressure]=dataset[pressure]*(1.0+((gamma*z_0)/(dataset[temperature]-z_0*gamma)))**(-g/(gamma*r_d))
    return dataset

##########################################################
def density_calculator(dataset,z_0,temperature,pressure,humidity,r_d=287.0,epsilon=0.622,e_star=611.0,L=2.26*10**6,r_v=461.0,T_star=273.15):
    # This function firstly changes the unit for
    # temperature from centigrade to kelvin, scale of relative humidity from (0-100) to (0,1) and unit of pressure as milibar to pascal. Also,
    # it extrapolates the pressure to calculate pressure at hub height, assuming lapse rate is constant. Then the density is calculated. 
    df=dataset[[temperature,pressure,humidity]]
    
    df[temperature]=df[temperature]+273.15
    df[humidity]=0.01*df[humidity]
    df[pressure]=100.0*df[pressure]
    df=pressure_extrapolator(df,z_0,pressure,temperature)  

    df['r (kg/kg)']=((epsilon*e_star*(df[humidity]/df[pressure])*np.exp((L/r_v)*((1/T_star)-(1/df[temperature]))))**(-1.0)-1.0)**(-1.0)
    df['rho (kg/m^3)']=df[pressure]/(r_d*df[temperature]*(1+0.61*df['r (kg/kg)']))

    return df[['rho (kg/m^3)']]    

##########################################################
def null_value_counter(dataset,missing_df,qa_step,var):
    # this function counts numeric value of our dataset after each QA step for all variables
    for column in var:
        missing_df.loc[qa_step,column]=dataset[column].isnull().value_counts().loc[True]
        
    return missing_df
##########################################################
def null_percent_calculator(dataset):
    # this function takes the dataset,containing number of null values and calculate percentage of null values and return it.  
    for column in dataset.columns:
        x=dataset[column].iloc[1:].dropna().diff(1)/dataset.loc['numeric values',column]
        x=100*x
    
        for name in x.index:
            dataset.loc[name,column]=x.loc[name]
        
        dataset.loc['total',column]=dataset[column].iloc[1:].sum()
       
    dataset.drop(index=['numeric values','null values'],axis=1,inplace=True)
    
    return dataset

##########################################################