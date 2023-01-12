### code developer: Mohammad Ghayuri
### Email: mohammadghayuri@gmail.com

# this code performs the quality assurance (QA) methodology, indtroduced in the paper entitled as
# "Anomaly Detection and treatment for Meteorological and Wind Turbine Power Measurements". 

# First utiliy module should be run and then each of the following steps can be employed. Each step has several parts, combining the 
# analyst's judgment to make a decision about critical values, according to the relevant figures and then perform the next part of each 
# part for each step. 

#########################################
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime,date,timedelta
from pyod.models.knn import KNN
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import CubicSpline

from utility import binned_average, block_identifier, confidence_interval, daily_gust, density_calculator, duplicates_date_identifier,\
 extreme_variation_identifier, fixed_limit, icing, inconsistency_remover, mast_interference, moving_mean_std, null_percent_calculator,\
 null_value_counter, outlier_detector, plotter, pressure_extrapolator, repeated_constant_remover, small_outlier_remover, transfromer,\
turbine_wake_angle_detector

########################################################## 
# This part set the values of parameters. 
"""
# the parameters in this part are used for pressure extrapolation and calculation of density. Their values are directly used in the function
# in the utility module and their presence in the main file is not necessary. Just for giving information about their valuse, they are here. 
    # epsilon is ...
    # r_d and r_v are respectively ...
    # L is the latent heat
    # e_star is ...
    # T_star is zero temperature in kelvin
    # gamma is the lapse rate
    # g is gravity accelaration.   
r_d=287.0
epsilon=0.622
e_star=611.0
L=2.26*10**6
r_v=461.0
T_star=273.15
gamma=-0.006
g=9.8
"""
# z_0 is the height of turbine (hub height). the hub height is originally 75m but since we need that to convert pressure at 75m using its value
# at 1m, we consider the one-meter height as the origin of the height. So in our code that's 74m instead of 75m.
z_0=74.0
# I've set the following date to remove biased values, detected in moving average step. 
# TIME_0 is starting date after calibration and TIME_1, TIME_2 & TIME_3 respectively are ending date for u3, phi2 and rh2 after which
# their values are biased. 

TIME_0='2017-12-31'
TIME_1='2018-11-20'
TIME_2='2020-03-31'
TIME_3='2019-01-28'

########################
# QA-step3 (Icing effect)
t='t1'
t_c=2.0
u_c=1.2
sonic_u='u5'
cup_u=['u1','u2']
cup_u_prime=['u3','u4']

########################
### QA-step4 (Fixed limit check)
# the following parameters specifies lower and higher bound for minimum and maximum values for temperature, pressure, relative humidity
# and wind direction to remove the extreme values. wd_interval is for splitting data into several categories based on direction group
# since I want to removes the south eastern data, considering the location of T5 and met mast. 
P_0=0.0
P_1=2050.0

U_0=0.0
U_1=35.0  

PHI_0=0.0
PHI_1=360.0

T_0=-24.0
T_1=37.0

BP_0=955.0
BP_1=1040.0

RH_0=0.0
RH_1=100.0
   
limits={('p'):(P_0,P_1),('u1','u2','u3','u4','u5'):(U_0,U_1),('phi1','phi2','phi3'):(PHI_0,PHI_1),('t1','t2'):(T_0,T_1)\
,('rh1','rh2'):(RH_0,RH_1),('bp1','bp2'):(BP_0,BP_1)}

########################
# dirty_df is the dataset before any cleaning process. the only one is the process we removed the data measured at any time
# except 10 min interval, which was done in unifier code. 
dirty_df=pd.read_csv(input_path)

dirty_df=dirty_df[['time.10min', 't5.power.10min (kw)', 'ws1.10min (m/s)', 'ws2.10min (m/s)',
'ws3.10min (m/s)', 'ws4.10min (m/s)', 'ws5.10min (m/s)', 'wd1.10min (degree)', 'wd2.10min (degree)', 'wd3.10min (degree)',
'temp1.10min (c)', 'temp2.10min (c)', 'press1.10min (mb)', 'press2.10min (mb)', 'rh1.10min (%)', 'rh2.10min (%)']]

dirty_df=dirty_df.rename(columns={'time.10min':'time', 't5.power.10min (kw)':'p', 'ws1.10min (m/s)':'u1', 'ws2.10min (m/s)':'u2',
'ws3.10min (m/s)':'u3', 'ws4.10min (m/s)':'u4', 'ws5.10min (m/s)':'u5', 'wd1.10min (degree)':'phi1', 'wd2.10min (degree)':'phi2', 
'wd3.10min (degree)':'phi3',
'temp1.10min (c)':'t1', 'temp2.10min (c)':'t2', 'press1.10min (mb)':'bp1', 'press2.10min (mb)':'bp2', 'rh1.10min (%)':'rh1', 'rh2.10min (%)':'rh2'})

dirty_df['time']=pd.to_datetime(dirty_df['time'])
dirty_df.set_index('time',inplace=True)

###########################
###########################
# QA-step1 (Bias detection)
moving_mean,moving_std=moving_mean_std(dirty_df)

"""
# part I
# This part of the code plots moving average and standard deviation. You can plot the moving mean and std to see their behavior, looking for
# any bias in the dataset. 
plt.figure()
plotter(moving_mean,'u1',xlabel='time',ylabel='u1',color='grey',label='moving mean')
plotter(moving_std,'u1',xlabel='time',ylabel='u1',color='black',label='moving std')
plt.show()
"""
# part II
# In this part biased values, detected, are manually removed. 
dirty_df=dirty_df[dirty_df.index.date>pd.to_datetime(TIME_0)]

###############################
# missing_df counts missing values, null, and collect them in a dataframe to show how many null are added after each step. Before
# perfoming the first step we we create missing_df to see how many missing values exists before implementation of the first step of our 
# QA, in the original dataset provided by data provider, in this case WEICAN. 

missing_df=pd.DataFrame(index=['numeric values','null values','moving average','wake effect of turbine','icing','fixed limit',\
'repeated constant values','confidence interval','abanormally high variation','interference by the mast',\
'distinguishing outliers from true small values','total'], columns=dirty_df.columns)
for column in missing_df.columns:
    missing_df.loc['numeric values',column]=dirty_df[column].isnull().value_counts().loc[False]
    missing_df.loc['null values',column]=dirty_df[column].isnull().value_counts().loc[True]
    
###############################
# part III
    
dirty_df.loc[dirty_df.index.date>pd.to_datetime(TIME_1),'u3']=np.nan
dirty_df.loc[dirty_df.index.date>pd.to_datetime(TIME_2),'phi2']=np.nan
dirty_df.loc[dirty_df.index.date>pd.to_datetime(TIME_3),'rh2']=np.nan

###############################
# this part counts the number of numeric value after removing the biased values for three variables: 'u3','phi2','rh2'. for the next
# step we follow the similar procedure
qa_step='moving average'
var=['u3','phi2','rh2']
missing_df=null_value_counter(dirty_df,missing_df,qa_step,var)

####################
# QA-step2 ( Wake effect of neighboring wind turbine)
# In part I, we do some experiences to see wind directions for which wind speed, measured at the mast, are affected by interference of 
# neighboring wind turbine. After the identification of these boundaries for wind direction, in part II we replace
# the affected values with NAN
# part I
manu_u=np.arange(1.0, 26.0, 1.0)
manu_p=np.array([0.00, 0.00, 0.00, 0.00, 93.24, 251.75, 503.50, 862.47, 1249.42, 1641.03, 1888.11, 2000.00, 2000.00, 2000.00, 2000.00, 2000.00, 2000.00, 2000.00, 2000.00,	2000.00, 2000.00, 2000.00, 2000.00, 2000.00, 2000.00])
manu_power_curve=CubicSpline(manu_u,manu_p,bc_type='clamped')
inverse_manu_power_curve=CubicSpline(manu_p[3:12],manu_u[3:12],bc_type='natural')

ws_list=['u1', 'u2', 'u3', 'u4', 'u5']
PHI_PRIME_0=90.0
PHI_PRIME_1=180.0
DELTA_PHI=3
rated_u=11.0
phi='phi1'
p='p'
interval=100

"""
for u in ws_list:
    deviation_df,df=turbine_wake_angle_detector(dirty_df,phi,PHI_PRIME_0,PHI_PRIME_1,DELTA_PHI,inverse_manu_power_curve,rated_u,u,p,interval,power_min=0,power_max=1950)
    plt.plot(deviation_df['wind direction'],deviation_df['ws deviation'],label=u)
    plt.legend()
    plt.show()
"""
# part II
PHI_PRIME_2=130
PHI_PRIME_3=170

"""
plotter(dirty_df.loc[(dirty_df[phi]>=PHI_PRIME_2)&(dirty_df[phi]<PHI_PRIME_3)],x='u5',y='p',xlabel='u5 (m/s)',ylabel='p (kw)',color='black',size=2)
plotter(dirty_df.loc[(dirty_df[phi]<PHI_PRIME_2)|(dirty_df[phi]>=PHI_PRIME_3)],x='u5',y='p',xlabel='u5 (m/s)',ylabel='p (kw)',color='grey',size=2)
"""
for u in ws_list:
    dirty_df.loc[(dirty_df[phi]>=PHI_PRIME_2)&(dirty_df[phi]<PHI_PRIME_3),u]=np.nan
   
###############################
qa_step='wake effect of turbine'
var=dirty_df.columns
missing_df=null_value_counter(dirty_df,missing_df,qa_step,var)    
    
###############################        
# QA-step3 (Icing effect)
# this part identifies wrong wind speed measurements and replace them with NAN. 

"""
# part I

# In this part you can plot wind speeds measured by sonic anemometer, sonic_u, and cup anemometers, cup_u, as well as  their differences 
# for t>t_c when icing is not probable and decide critical differnce for which higher values are considered uncommon, resulting from icing. 
# In our cases, after experiences our critical difference is 1.2 m/s. 
plotter(dataset=dirty_df,x='u1',y='u5',size=2,xlabel='u1',ylabel='u5')

u=cup_u[0]
ice_df=pd.DataFrame(dirty_df.loc[(dirty_df[t]>t_c),sonic_u]-dirty_df.loc[(dirty_df[t]>t_c),u],columns=['delta_u'])
plotter(dataset=ice_df,x='delta_u',bins=300,xlim=[0,3],ylim=[0,100])
"""
# part II
dirty_df=icing(dirty_df, t, t_c, u_c, sonic_u, cup_u, cup_u_prime)
###############################
qa_step='icing'
var=['u1','u2','u3','u4']
missing_df=null_value_counter(dirty_df,missing_df,qa_step,var)

###############################                         
### QA-step4 (Fixed limit check)
# This section using fixed limit to restrict the values of variables. it removes non physical and extreme values and replace them with NaN.     
dirty_df=fixed_limit(dirty_df,limits)

###########################
qa_step='fixed limit'
var=dirty_df.columns
missing_df=null_value_counter(dirty_df,missing_df,qa_step,var)

###############################        
        
### QA-step5 (Repeated constant values)
## In these two parts, we do some experiences to determine critival block length for each variable for part III, next section.  
"""
# part I
# In this part, using block_identifier function, "duplicate hist" shows block index and its length. Using plotter function you plot 
# the histogram to see frequency for different block length. 
target_var='u3'
duplicate_hist=block_identifier(dirty_df,target_var)
plotter(duplicate_hist,x='block length',bins=50,ylim=[0,3],xlabel='block length',ylabel='frequency')

# part II
# In the following part, "duplicates_date_identifier" function finds the dates for which we have block length>critical block length. 
# Then we plot target var vs counterpart var for those dates, seeking for any abnormal behavior 

critical_block_length=1
duplicates_dates=duplicates_date_identifier(dirty_df,target_var,critical_block_length)
counterpart_var='u5'
plotter(dataset=dirty_df.loc[duplicates_dates,[target_var,counterpart_var]],x=target_var,y=counterpart_var,xlabel=target_var,ylabel=counterpart_var,size=10)
"""
######################
# part III
## in this part we use the critical values determined in experiences in part I and II, for each variable in the following dictionary,
## to remove the repeated constant values which represent wrong data.
     
critical_block_lengths={'p':1, 'u1':2, 'u2':2, 'u3':2, 'u4':2, 'u5':2, 'phi1':2, 'phi2':2, 'phi3':2, 't1':6, 't2':5, 'bp1':100, 'bp2':60, 'rh1':2, 'rh2':2}
dirty_df=repeated_constant_remover(dirty_df,critical_block_lengths)
   
###########################
qa_step='repeated constant values'
var=dirty_df.columns
missing_df=null_value_counter(dirty_df,missing_df,qa_step,var)

###############################  
### QA-step6 (Confidence interval)     
# in part I we do some experiences and plot target variable vs counterpart one to find appropriate lower and upper limit beyond which
# points are considered internally incosistent. 
"""
# part I

target_var='u1'
counterpart_var='u5'
division_number=100
lower_percentage=0.5
higher_percentage=99.5
    
percentiles,suspicous_date=confidence_interval(dirty_df,target_var,counterpart_var,division_number,lower_percentage,higher_percentage)   
plotter(dirty_df.loc[suspicous_date],x=target_var,y=counterpart_var,size=10,color='grey')
plotter(percentiles,x=target_var,y='lower percentile',color='black')
plotter(percentiles,x=target_var,y='median',color='black')
plotter(percentiles,x=target_var,y='higher percentile',color='black',xlabel=target_var,ylabel=counterpart_var)
plt.show()
"""

# part II
# After finding appropriate upper and lower limits, we set them as the boundar of confidence interval in the conf_dic and replace 
# values beyond these limits with NaN. 

division_number=100
"""
conf_dic={'u1':['u5',0.5,99.5], 'u2':['u5',0.5,99.5], 'u3':['u5',0.5,99.5], 'u4':['u5',0.5,99.5], 'phi1':['phi3',0.5,99.5], \
'phi2':['phi3',0.5,99.5], 't1':['t2',0.5,99.5], 'bp1':['bp2',0.5,99.5], 'rh1':['rh2',0.5,99.5],'u1':['p',0.5,99.5],'u2':['p',0.5,99.5],\
'u5':['p',0.5,99.5]}
"""
conf_dic={'u1':['u5',0.5,99.5], 'u2':['u5',0.5,99.5], 'u3':['u5',0.5,99.5], 'u4':['u5',0.5,99.5], 'phi1':['phi3',0.5,99.5], \
'phi2':['phi3',0.5,99.5], 't1':['t2',0.5,99.5], 'bp1':['bp2',0.5,99.5], 'rh1':['rh2',0.5,99.5],'u1':['p',1.5,99.0],'u2':['p',1.5,99.0],\
'u5':['p',1.5,99.0]}

dirty_df=inconsistency_remover(dirty_df,conf_dic,division_number)

##########################
# part III

# wind power, in addition to the wind speed, depends on the density too. So we split each wind speed bin into several 
# groups, based on density values, and use confidence interval for outlier detection.
df=density_calculator(dirty_df,z_0,temperature='t1',pressure='bp1',humidity='rh1')
dirty_df=pd.concat([dirty_df,df],axis=1)

density=np.linspace(dirty_df['rho (kg/m^3)'].min(),dirty_df['rho (kg/m^3)'].max(),10)
var_1='u5'
var_2='p'
division_number=100
lower_percentage=0.5
higher_percentage=99.5
v_cut_in=4.0
v_rated=12.0
for u in ['u1','u2','u5']:
    for i in range(len(density)-1):
        percentiles,suspicous_date=confidence_interval(dirty_df[(dirty_df[var_1]>v_cut_in)&(dirty_df[var_1]<=v_rated)&(dirty_df['rho (kg/m^3)']>=density[i])&(dirty_df['rho (kg/m^3)']<density[i+1])] ,var_1,var_2,division_number,lower_percentage,higher_percentage)
        dirty_df.loc[suspicous_date,[var_2]]=np.nan 
    
############################### 
qa_step='confidence interval'
var=dirty_df.columns
missing_df=null_value_counter(dirty_df,missing_df,qa_step,var)

###########################
### QA-step7 (Abnormally high variation)
# in part I, we plot histogram, showing frequency vs the differences between two consecutive steps for each variable to judge what 
# is critical value beyond which the difference between two consecutive steps is considered extreme variation. In part II we set such
# limitation to replace the values which are abnormally jumped or dropped. 
"""
# part I

var='u1'
variation_df=pd.DataFrame(np.abs(dirty_df[var].diff()))
plotter(dataset=variation_df,x=var,xlabel='the difference between two consecutive steps',ylabel='frequency',ylim=[0,20],bins=20)
"""
####################
# part II
extreme_dic={'p':1300.0, 'u1':6.0, 'u2':6.0, 'u3':5.0, 'u4':5.0, 'u5':6.0, 't1':4.0, 't2':4.0, 'bp1':1.5, 'bp2':1.5, 'rh1':15.0, 'rh2':15.0}
dirty_df=extreme_variation_identifier(dirty_df,extreme_dic)

####################
### QA-step7 (Abnormally high variation - gust analysis) 
# in part I we calculate daily mean wind speed and the gust factor and plot their logarithm, seeking for any deviation from linear
# relationship. Then for the day, represented by those points, we plot the daily time series for any extreme jump and drop for each variable. 
# in part III we replace that jump or drop with NaN.  
"""
# part I
var='u5'
total_number=144.0
not_null_percentage=0.5
not_null_number=total_number*not_null_percentage
outliers_fraction=0.02
minimum_anomaly_score=0.0005
cleaner_model=KNN(method='mean',contamination=outliers_fraction)

gust_df=daily_gust(dirty_df,var,not_null_number)
inliers,outliers=outlier_detector(gust_df,cleaner_model,outliers_fraction,minimum_anomaly_score)

plotter(dataset=inliers,x='ln u',y='ln g',color='grey',size=12)
plotter(dataset=outliers,x='ln u',y='ln g',color='black',size=12,xlabel='ln U',ylabel='ln G')

# part II
i=0
for date in outliers.index:
    df=dirty_df.loc[pd.date_range(date,date+timedelta(hours=24),freq='10min')]
    df.index.name='time'
    df.reset_index(inplace=True)
    plotter(dataset=df,x=var,ylabel=var,marker='o')
    plt.savefig(output_path)
    plt.show()
    i+=1
""" 
# part III
# After examining the daily time series of suspicious dates, we can insert the dates with abnormall jumps or drops in the following dictionary
# to remove the relevant wind speed. 
jumps_dic_dates={'u2':[pd.to_datetime('2019-07-21 02:00:00')]}

for key in jumps_dic_dates.keys():
    for d in jumps_dic_dates[key]:
        dirty_df.loc[d,key]=np.nan
        
###########################
qa_step='abanormally high variation'
var=dirty_df.columns
missing_df=null_value_counter(dirty_df,missing_df,qa_step,var)

###############################  
### QA-step8 (Interference by the mast) 
# in part I, firstly we plot ratio of wind speed measured by different sensors on the mast and plot them vs wind direction to see 
# which direction is affected by the mast, via comparision beween two values. After seeing the diagrams, in part II, we decide boundaries
# of wind direction, affected by the mast to replace the the values inside the the boundaries with NaN. 
"""
# part I
u_var1='u5'
u_var2='u2'
phi_var='phi3'
division_number=100
lower_percentage=0.5
higher_percentage=99.5
u_ratio=u_var1+'/'+u_var2

df=dirty_df[[phi_var]]
df[u_ratio]=dirty_df[u_var1]/dirty_df[u_var2]
percentiles,suspicous_date=confidence_interval(df,phi_var,u_ratio,division_number,lower_percentage,higher_percentage)


plotter(dataset=df,x=phi_var,y=u_ratio,color='grey',ylim=[0.9,1.15],size=2)
plotter(dataset=percentiles,x=phi_var,y='median',xlabel=phi_var,ylabel=u_ratio,color='black')
plt.show()
"""

# part II
# this part finds the wind speed values affected by wake effect caused by the tower and and replace them with NaN.     
tower_wake_dic={'u1':[314,340],'u2':[160,200],'u5':[335,360]}
phi='phi3'
dirty_df=mast_interference(dirty_df,tower_wake_dic,phi)
  
###########################
qa_step='interference by the mast'
var=['u1','u2','u5']
missing_df=null_value_counter(dirty_df,missing_df,qa_step,var)

############################### 
### QA-step9 ( Distinguishing outliers from true small values)
# in part I we plot time series and histogram for transformed form of each variable, based on equation 4 in the paper, seeking for
# any small values. according to histogram, in part II, we set boundaries to for which higher x_hat or smaller x is considered suspicious
# and plot them vs counterpart variable (i.e. u1 and u5) 
"""
# part I
var='u1'
transformed_df=transfromer(dirty_df)

plotter(dataset=transformed_df,x=var)
# to plot histogram use the following code as we have infinity because of minimum value in each column.
plotter(dataset=transformed_df[np.isfinite(transformed_df[var])],x=var,bins=50,ylim=[0,50])
"""

# part II
low_dic={'p':['u5',100.0], 'u1':['u5',30.0], 'u2':['u5',30.0], 'u3':['u5',20.0], 'u4':['u5',30.0], 'u5':['u1',30.0], 
'phi1':['phi3',100.0], 'phi2':['phi3',100.0], 'phi3':['phi1',100.0], 't1':['t2',12.0], 't2':['t1',12.0], 'bp1':['bp2',30.0], 
'bp2':['bp1',50.0], 'rh1':['rh2',5.0], 'rh2':['rh1',4.0]}

lower_percentage=5.0
higher_percentage=95.0
division_number=100.0

dirty_df=small_outlier_remover(dirty_df,low_dic,lower_percentage,higher_percentage,division_number)

###########################
qa_step='distinguishing outliers from true small values'
var=dirty_df.columns
missing_df=null_value_counter(dirty_df,missing_df,qa_step,var)

############################### 
# This part calculates and show percentage of null values after each step of QA methodology in the paper.   
missing_df=null_percent_calculator(missing_df)

###############################
# this part saves the cleaned dataset into the following path. 
dirty_df.to_csv(output_path)

