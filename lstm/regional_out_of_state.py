## EVALUATE REGIONAL LSTM MODEL ON OUT-OF-STATE SITES ##

# preliminaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import xarray as xr
import random as rn
import pickle
from captum.attr import IntegratedGradients
import traceback

from _lstm import *
from _data import *

## GET INFORMATION FROM USER ##
# inputs are run name, state can also be made an input but is not currently
run_name = input("run name: ")
#state = input('state: ')
state_list = ['CA','WA','UT','ID']
total_statistics = pd.DataFrame(columns=['state','rmse', 'normal rmse', 'nse', 'r2', 'spearman_rho', 'delta peak', 'normal delta peak', 'delta days'])

## EVALUATE MODEL FOR ALL STATES IN LIST ##
for k in range(0, len(state_list)):
    state = state_list[k]
    print(state)
    
    ## GET SITE DATA ##
    num_sites = 24
    num_years = 3
    lstm = torch.load('output/'+run_name+'_lstm.pt', map_location = DEVICE)
    with open('output/'+run_name + '_normalize.pkl', 'rb') as file:  
        l_normalize = pickle.load(file)
    scaler_swe = l_normalize[0]
    
    snotel = get_sites_random(state, num_sites)
    data = pd.DataFrame(columns=['site_id', 'year','train'])
    
    for i in range(0, len(snotel)):
        site_id = snotel['site_id'][i]
        start_date = snotel['first_wy_date'][i]
        end_date = snotel['last_wy_date'][i]
        try:
            with redirect_stdout(trap):
                years = get_years_precip(start_date, end_date, site_id, num_years) 
    
            for j in range(0, len(years)):
                data.loc[len(data.index)] = [site_id, int(years[j]), False] 
        except:
            print('missing data for ', site_id)
            traceback.print_exc()
    
    ## GET YEARLY DATA ##
    l_swe_test = [] 
    l_non_swe_test = []
    
    data_test = data.loc[(data.loc[:,'train'] == False)].reset_index().drop(columns=['index','train'])
    
    for j in range(0, len(data_test)):
        site_id = data_test['site_id'][j]
        year = data_test['year'][j]
        start_date = str(year-1) + '-10-01'
        end_date = str(year) + '-09-30'
        try:
            with redirect_stdout(trap):
                swe, non_swe = get_sc_data(site_id, start_date, end_date)
            l_swe_test.append(swe)
            l_non_swe_test.append(non_swe)
    
            # add site data
            data_test.loc[j, 'latitude'] = non_swe.loc[0,'latitude']
            data_test.loc[j, 'longitude'] = non_swe.loc[0,'longitude']
            data_test.loc[j, 'elevation'] = non_swe.loc[0,'elevation']
            data_test.loc[j, 'land cover'] = non_swe.loc[0,'land_cover']
            data_test.loc[j, 'slope_x'] = non_swe.loc[0,'slope_x']
            data_test.loc[j, 'slope_y'] = non_swe.loc[0,'slope_y']
        except:
            print('missing data for ', site_id, " : ", year)
            data_test = data_test.drop(j)
            traceback.print_exc()
    
    data_test = data_test.reset_index()
    
    ## MODEL INPUT
    test_swe = pd.concat(l_swe_test).reset_index().drop(columns='index')
    test_non_swe = pd.concat(l_non_swe_test).reset_index().drop(columns='index')
    test_swe_tensors, test_non_swe_tensors, test_sites, test_years = create_dataset(test_swe, test_non_swe, l_normalize)
    #test_swe_tensors = test_swe_tensors.to(DEVICE)
    #test_non_swe_tensors = test_non_swe_tensors.to(DEVICE)
    
    ## ANALYZE MODEL
    statistics, feature_importance = analyze_results(lstm, data_test, test_swe_tensors, test_non_swe_tensors, scaler_swe, False)
    print('statistics for: ' + run_name + state)
    print(f"RMSE: {np.mean(statistics['rmse']):.2f}")
    print(f"normal RMSE: {np.mean(statistics['normal rmse']):.2f}")
    print(f"NSE: {np.mean(statistics['nse']):.2f}")
    print(f"R2: {np.mean(statistics['r2']):.2f}")
    print(f"Spearman's rho: {np.mean(statistics['spearman_rho']):.2f}")
    print(f"delta peak SWE: {np.mean(statistics['delta peak']):.2f}")
    print(f"normal delta peak SWE: {np.mean(statistics['normal delta peak']):.2f}")
    print(f"delta days: {np.mean(statistics['delta days']):.2f}")

    ## SAVE OUTPUT ##
    statistics.to_csv('output/out_of_state/'+run_name+'_'+state+'_statistics.txt',sep=' ',header=None, index=False, index_label=False)
    
    total_statistics.loc[len(total_statistics)] = [state, np.mean(statistics['rmse']), np.mean(statistics['normal rmse']), np.mean(statistics['nse']),
                                                   np.mean(statistics['r2']), np.mean(statistics['spearman_rho']), np.mean(statistics['delta peak']),
                                                   np.mean(statistics['normal delta peak']), np.mean(statistics['delta days'])]

total_statistics.to_csv('output/out_of_state/'+run_name+'_statistics.txt',sep=' ',header=None, index=False, index_label=False)