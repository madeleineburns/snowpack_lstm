# functions for managing data in LSTM training and analysis process

## PRELIMINARIES ##
# general 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import xarray as xr
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy import stats
import random as rn
from captum.attr import IntegratedGradients

# for data
import subsettools as st
import hf_hydrodata as hf
from pandas.tseries.offsets import DateOffset

from contextlib import redirect_stdout
trap = io.StringIO()

from _lstm import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

## WATER YEAR START ##
# given datetime date, return start of next water year
# first year where precip, swe, temp data are available
def get_wy_start(date, site_id):
    try:
        with redirect_stdout(trap):
            site_df_swe = hf.get_point_data(dataset='snotel', variable='swe', temporal_resolution='daily', aggregation='sod', site_ids=[site_id])
            site_df_precip = hf.get_point_data(dataset='snotel', variable='precipitation', temporal_resolution='daily', aggregation='sum', site_ids=[site_id])
            site_df_temp = hf.get_point_data(dataset='snotel', variable='air_temp', temporal_resolution='daily', aggregation='mean', site_ids=[site_id])
    except:
        return np.nan

    date = max([pd.to_datetime(site_df_swe['date'][0]), pd.to_datetime(site_df_temp['date'][0]), pd.to_datetime(site_df_precip['date'][0])])
    
    if (date.month == 10) and (date.day == 1):
        data_start = date.strftime('%Y-%m-%d')
    elif date.month < 10:
        data_start = str(date.year)+'-10-01'
    else:
        data_start = str(date.year+1)+'-10-01'
    return data_start

## WATER YEAR END ##
# given datetime date, return end of next water year
# compatible with CW3E (ends on WY 2022)
def get_wy_end(date, site_id):
    try:
        with redirect_stdout(trap):
            site_df_swe = hf.get_point_data(dataset='snotel', variable='swe', temporal_resolution='daily', aggregation='sod', site_ids=[site_id])
            site_df_precip = hf.get_point_data(dataset='snotel', variable='precipitation', temporal_resolution='daily', aggregation='sum', site_ids=[site_id])
            site_df_temp = hf.get_point_data(dataset='snotel', variable='air_temp', temporal_resolution='daily', aggregation='mean', site_ids=[site_id])
    except:
        return np.nan

    date = min([pd.to_datetime(site_df_swe['date'][len(site_df_swe)-1]), pd.to_datetime(site_df_temp['date'][len(site_df_temp)-1]), 
            pd.to_datetime(site_df_precip['date'][len(site_df_precip)-1])])
    
    if (date.month == 9) and (date.day == 30):
        data_end = date.strftime('%Y-%m-%d')
    elif date.month < 10:
        data_end = str(date.year-1)+'-09-30'
    else:
        data_end = str(date.year)+'-09-30'
    if(date.year == 2023) or (date.year == 2024):
        data_end = '2022-09-30'
    return data_end


## GET ALL SNOTEL SITES ##
# for all states
def get_sites_full(num_sites):
    snotel = hf.get_site_variables(variable="swe")
    snotel = snotel.reset_index().drop(columns=['index','variable_name','units','site_query_url','date_metadata_last_updated','tz_cd','doi'])

    # read in testing data and remove it from dataset
    data_test = pd.read_csv('national_test_sites.txt', sep=' ',header=None)
    data_test.columns = ['site_id', 'site_name', 'site_type', 'agency', 'state','first_date_data_available', 'last_date_data_available', 'record_count',
                         'latitude', 'longitude', 'bins', 'first_wy_date', 'last_wy_date']
    data_test = data_test.drop(columns=['bins', 'first_wy_date', 'last_wy_date'])
    snotel = pd.merge(snotel, data_test, on=['site_id','site_name','site_type','agency','state','first_date_data_available',
                                                       'last_date_data_available', 'record_count','latitude','longitude'], 
                           how='outer', indicator=True).query("_merge != 'both'").drop('_merge', axis=1).reset_index(drop=True)
    
    snotel['first_wy_date'] = snotel.apply(lambda x: get_wy_start(pd.to_datetime(x.first_date_data_available), x.site_id), axis=1)
    snotel['last_wy_date'] = snotel.apply(lambda x: get_wy_end(pd.to_datetime(x.last_date_data_available), x.site_id), axis=1)
    snotel = snotel.dropna(axis='index', how='any').reset_index()

    return snotel

## GET ALL SNOTEL SITES ##
# for given state
def get_sites_full_state(state, num_sites):
    snotel = hf.get_site_variables(variable="swe", state=state)
    snotel = snotel.reset_index().drop(columns=['index','variable_name','units','site_query_url','date_metadata_last_updated','tz_cd','doi'])

    # read in testing data and remove it from dataset
    data_test = pd.read_csv('national_test_sites.txt', sep=' ',header=None)
    data_test.columns = ['site_id', 'site_name', 'site_type', 'agency', 'state','first_date_data_available', 'last_date_data_available', 'record_count',
                         'latitude', 'longitude', 'bins', 'first_wy_date', 'last_wy_date']
    data_test = data_test.drop(columns=['bins', 'first_wy_date', 'last_wy_date'])
    snotel = pd.merge(snotel, data_test, on=['site_id','site_name','site_type','agency','state','first_date_data_available',
                                                       'last_date_data_available', 'record_count','latitude','longitude'], 
                           how='outer', indicator=True).query("_merge != 'both'").drop('_merge', axis=1).reset_index(drop=True)
    
    snotel['first_wy_date'] = snotel.apply(lambda x: get_wy_start(pd.to_datetime(x.first_date_data_available), x.site_id), axis=1)
    snotel['last_wy_date'] = snotel.apply(lambda x: get_wy_end(pd.to_datetime(x.last_date_data_available), x.site_id), axis=1)
    snotel = snotel.dropna(axis='index', how='any').reset_index()

    return snotel


## GET RANDOM SELECTION OF SNOTEL SITES ##
# for all states and given number of sites
def get_sites_random(num_sites):
    snotel_full = hf.get_site_variables(variable="swe")
    snotel_full = snotel_full.reset_index().drop(columns=['index','variable_name','units','site_query_url','date_metadata_last_updated','tz_cd','doi'])

    # read in testing data and remove it from dataset
    data_test = pd.read_csv('national_test_sites.txt', sep=' ',header=None)
    data_test.columns = ['site_id', 'site_name', 'site_type', 'agency', 'state','first_date_data_available', 'last_date_data_available', 'record_count',
                         'latitude', 'longitude', 'bins', 'first_wy_date', 'last_wy_date']
    data_test = data_test.drop(columns=['bins', 'first_wy_date', 'last_wy_date'])
    snotel_full = pd.merge(snotel_full, data_test, on=['site_id','site_name','site_type','agency','state','first_date_data_available',
                                                       'last_date_data_available', 'record_count','latitude','longitude'], 
                           how='outer', indicator=True).query("_merge != 'both'").drop('_merge', axis=1).reset_index(drop=True)

    if(num_sites > len(snotel_full)):
        return snotel

    # select a random selection of num_sites from snotel data
    #rn.seed(8)
    sample = rn.sample(list(snotel_full.index), num_sites)
    snotel = snotel_full.iloc[sample].reset_index()
    
    snotel['first_wy_date'] = snotel.apply(lambda x: get_wy_start(pd.to_datetime(x.first_date_data_available), x.site_id), axis=1)
    snotel['last_wy_date'] = snotel.apply(lambda x: get_wy_end(pd.to_datetime(x.last_date_data_available), x.site_id), axis=1)
    snotel = snotel.dropna(axis='index', how='any').reset_index()

    return snotel


## GET RANDOM SELECTION OF SNOTEL SITES ##
# for given state and number of sites
def get_sites_random_state(state, num_sites):
    snotel = hf.get_site_variables(variable="swe", state=state)
    snotel_full = snotel_full.reset_index().drop(columns=['index','variable_name','units','site_query_url','date_metadata_last_updated','tz_cd','doi'])

    # read in testing data and remove it from dataset
    data_test = pd.read_csv('national_test_sites.txt', sep=' ',header=None)
    data_test.columns = ['site_id', 'site_name', 'site_type', 'agency', 'state','first_date_data_available', 'last_date_data_available', 'record_count',
                         'latitude', 'longitude', 'bins', 'first_wy_date', 'last_wy_date']
    data_test = data_test.drop(columns=['bins', 'first_wy_date', 'last_wy_date'])
    snotel_full = pd.merge(snotel_full, data_test, on=['site_id','site_name','site_type','agency','state','first_date_data_available',
                                                       'last_date_data_available', 'record_count','latitude','longitude'], 
                           how='outer', indicator=True).query("_merge != 'both'").drop('_merge', axis=1).reset_index(drop=True)

    if(num_sites > len(snotel_full)):
        return snotel

    # select a random selection of num_sites from snotel data
    #rn.seed(8)
    sample = rn.sample(list(snotel_full.index), num_sites)
    snotel = snotel_full.iloc[sample].reset_index()
    
    snotel['first_wy_date'] = snotel.apply(lambda x: get_wy_start(pd.to_datetime(x.first_date_data_available), x.site_id), axis=1)
    snotel['last_wy_date'] = snotel.apply(lambda x: get_wy_end(pd.to_datetime(x.last_date_data_available), x.site_id), axis=1)
    snotel = snotel.dropna(axis='index', how='any').reset_index()

    return snotel

    
## GET SELECTION OF SNOTEL SITES BASED ON LONGITUDE ONLY ##
# for given number of sites (divide by number of longitude bin, may not necessarily be accurate)
def get_sites_longitude(num_sites):  
    snotel_full = hf.get_site_variables(variable="swe")
    snotel_full = snotel_full.reset_index().drop(columns=['index','variable_name','units','site_query_url','date_metadata_last_updated','tz_cd','doi'])

    #read in testing data and remove it from dataset
    data_test = pd.read_csv('national_test_sites.txt', sep=' ',header=None)
    data_test.columns = ['site_id', 'site_name', 'site_type', 'agency', 'state','first_date_data_available', 'last_date_data_available', 'record_count',
                         'latitude', 'longitude', 'bins', 'first_wy_date', 'last_wy_date']
    data_test = data_test.drop(columns=['bins', 'first_wy_date', 'last_wy_date'])
    snotel_full = pd.merge(snotel_full, data_test, on=['site_id','site_name','site_type','agency','state','first_date_data_available',
                                                       'last_date_data_available', 'record_count','latitude','longitude'], 
                           how='outer', indicator=True).query("_merge != 'both'").drop('_merge', axis=1).reset_index(drop=True)
    
    #rn.seed(8)
    
    # FIRST: bin snotel sites into 3 based on longitude
    bins = [-125, -118, -111, -100]
    bin_labels=['maritime','intermountain','continental'] 
    snotel_full['bins'] = pd.cut(snotel_full['longitude'], bins=bins, labels=bin_labels, include_lowest=True)
    
    l_bins = []
    for b in bin_labels:
        size = int(np.ceil(num_sites/3))
        while size > 0:
            try:
                l_bins.extend(rn.sample(snotel_full.loc[snotel_full['bins'] == b].index.to_list(), size))
                break
            except:
                # if not enough sites exist, try to get mininum sample size
                size = size-1
                print('not enough sites in longitude ' + str(b) + '. try ' + str(size) +' sites.')
    snotel = snotel_full.iloc[l_bins].reset_index().drop(columns=['index'])
    snotel.loc[:,'first_wy_date'] = snotel.apply(lambda x: get_wy_start(pd.to_datetime(x.first_date_data_available), x.site_id), axis=1)
    snotel.loc[:,'last_wy_date'] = snotel.apply(lambda x: get_wy_end(pd.to_datetime(x.last_date_data_available), x.site_id), axis=1)
    snotel = snotel.dropna(axis='index', how='any')

    return snotel

## GET SELECTION OF SNOTEL SITES BASED ON LATITUDE and LONGITUDE ##
# for given number of sites (divide by number of latitude bin, may not necessarily be accurate)
def get_sites_latitude(num_sites):  
    snotel_full = hf.get_site_variables(variable="swe")
    snotel_full = snotel_full.reset_index().drop(columns=['index','variable_name','units','site_query_url','date_metadata_last_updated','tz_cd','doi'])

    # read in testing data and remove it from dataset
    data_test = pd.read_csv('national_test_sites.txt', sep=' ',header=None)
    data_test.columns = ['site_id', 'site_name', 'site_type', 'agency', 'state','first_date_data_available', 'last_date_data_available', 'record_count',
                         'latitude', 'longitude', 'bins', 'first_wy_date', 'last_wy_date']
    data_test = data_test.drop(columns=['bins', 'first_wy_date', 'last_wy_date'])
    snotel_full = pd.merge(snotel_full, data_test, on=['site_id','site_name','site_type','agency','state','first_date_data_available',
                                                       'last_date_data_available', 'record_count','latitude','longitude'], 
                           how='outer', indicator=True).query("_merge != 'both'").drop('_merge', axis=1).reset_index(drop=True)
    
    #rn.seed(8)
    
    # FIRST: bin snotel sites into 3 based on longitude
    bins = [-125, -118, -111, -100]
    bin_labels=['maritime','intermountain','continental'] 
    snotel_full['bins'] = pd.cut(snotel_full['longitude'], bins=bins, labels=bin_labels, include_lowest=True)
    
    l_bins = []
    for b in bin_labels:
        size = int(np.ceil(num_sites/3))
        while size > 0:
            try:
                l_bins.extend(rn.sample(snotel_full.loc[snotel_full['bins'] == b].index.to_list(), size))
                break
            except:
                # if not enough sites exist, try to get mininum sample size
                size = size-1
                print('not enough sites in longitude ' + str(b) + '. try ' + str(size) +' sites.')
    snotel_lon = snotel_full.iloc[l_bins].reset_index()
    
    # SECOND: bin snotel sites based on latitude - 18 bins
    bins = np.arange(32,51)
    snotel_lon['bins'] = pd.cut(snotel_lon['latitude'], bins=bins, labels=bins[:len(bins)-1], include_lowest=True)
    
    l_bins = []
    for b in bins[0:len(bins)-1]:
        size = int(np.ceil(num_sites/18))
        while size > 0:
            try:
                l_bins.extend(rn.sample(snotel_lon.loc[snotel_lon['bins'] == b].index.to_list(), size))
                break
            except:
                # if not enough sites exist, try to get mininum sample size
                size = size-1
                print('not enough sites in latitude ' + str(b) + '. try ' + str(size) +' sites.')
    
    snotel = snotel_lon.iloc[l_bins]
    snotel.loc[:,'first_wy_date'] = snotel.apply(lambda x: get_wy_start(pd.to_datetime(x.first_date_data_available), x.site_id), axis=1)
    snotel.loc[:,'last_wy_date'] = snotel.apply(lambda x: get_wy_end(pd.to_datetime(x.last_date_data_available), x.site_id), axis=1)
    snotel = snotel.dropna(axis='index', how='any').reset_index().drop(columns=['index','level_0'])

    return snotel


## GET SELECTION OF SNOTEL SITES BASED ON LATITUDE and LONGITUDE ##
# for given number of sites and state (divide by number of latitude bin, may not necessarily be accurate)
def get_sites_latitude_state(state, num_sites):  
    if num_sites < 8:
        print('number of sites must be greater than 8')
        return
        
    snotel_full = hf.get_site_variables(variable="swe", state='CO')
    snotel_full = snotel_full.reset_index().drop(columns=['index','variable_name','units','site_query_url','date_metadata_last_updated','tz_cd','doi'])

    # read in testing data and remove it from dataset
    data_test = pd.read_csv('regional_test_sites.txt', sep=' ',header=None)
    data_test.columns = ['site_id', 'site_name', 'site_type', 'agency', 'state','first_date_data_available', 'last_date_data_available', 'record_count',
                         'latitude', 'longitude', 'bins', 'first_wy_date', 'last_wy_date']
    data_test = data_test.drop(columns=['bins', 'first_wy_date', 'last_wy_date'])
    snotel_full = pd.merge(snotel_full, data_test, on=['site_id','site_name','site_type','agency','state','first_date_data_available',
                                                       'last_date_data_available', 'record_count','latitude','longitude'], 
                           how='outer', indicator=True).query("_merge != 'both'").drop('_merge', axis=1).reset_index(drop=True)
    
    #rn.seed(8)
    sample_size = int(num_sites/8)
    
    # bin snotel sites based on latitude
    bins = [37, 37.5, 38, 38.5, 39, 39.5, 40, 40.5, 41]
    snotel_full['bins'] = pd.cut(snotel_full['latitude'], bins=bins, labels=bins[:len(bins)-1], include_lowest=True)
    
    l_bins = []
    for b in bins[0:len(bins)-1]:
        size = sample_size
        while True:
            try:
                l_bins.extend(rn.sample(snotel_full.loc[snotel_full['bins'] == b].index.to_list(), size))
                break
            except:
                # if not enough sites exist, try to get mininum sample size
                size = size-1
                print('not enough sites in latitude ' + str(b) + '. try ' + str(size) +' sites.')
    
    snotel = snotel_full.iloc[l_bins]
    snotel.loc[:,'first_wy_date'] = snotel.apply(lambda x: get_wy_start(pd.to_datetime(x.first_date_data_available), x.site_id), axis=1)
    snotel.loc[:,'last_wy_date'] = snotel.apply(lambda x: get_wy_end(pd.to_datetime(x.last_date_data_available), x.site_id), axis=1)
    snotel = snotel.dropna(axis='index', how='any').reset_index()

    return snotel
    

## GET SINGLE COLUMN DATA ##
# returns SNOTEL and CW3E data for point location (snotel site with site_id)
# NOT formatted for input into LSTM: just a non-normalized 2D array
# assume start_date and end_date are water year adjusted
def get_sc_data(site_id, start_date, end_date):
    # GET SNOTEL DATA 
    site_df_swe = hf.get_point_data(dataset='snotel', variable='swe', temporal_resolution='daily', aggregation='sod', 
                                             date_start=start_date, date_end=end_date, site_ids=[site_id])
    site_df_precip = hf.get_point_data(dataset='snotel', variable='precipitation', temporal_resolution='daily', aggregation='sum', 
                                   date_start=start_date, date_end=end_date, site_ids=[site_id])
    site_df_temp = hf.get_point_data(dataset='snotel', variable='air_temp', temporal_resolution='daily', aggregation='mean', 
                                 date_start=start_date, date_end=end_date, site_ids=[site_id])
    metadata_df = hf.get_point_metadata(dataset='snotel', variable='swe', temporal_resolution='daily', aggregation='sod', 
                                             date_start=start_date, date_end=end_date, site_ids=[site_id])
    print('loaded SNOTEL data')
        
    # PARAMETERS FOR CW3E DATA
    precip = site_df_precip.set_axis(['date','precip'], axis='columns')
    temp = site_df_temp.set_axis(['date','temp'], axis='columns')
    tot_swe = site_df_swe.set_axis(['date','swe'], axis='columns')

    # adjust end date for CW3E
    end_date_cw3e = str(pd.to_datetime(end_date) + DateOffset(days=1))
    wy = str(pd.to_datetime(end_date).year)
    
    lat = metadata_df['latitude'][0]               	
    lon = metadata_df['longitude'][0]              
    bounds = st.latlon_to_ij([[lat, lon],[lat, lon]],"conus2")
    
    # GET DAILY CW3E DATA
    variables = ["precipitation", "downward_longwave", "downward_shortwave", "specific_humidity", "air_temp", "east_windspeed", "north_windspeed", 
                 "atmospheric_pressure"]
    if (not os.path.exists('scratch/'+site_id+'_'+wy+'.nc')):
        #os.remove('scratch/'+site_id+'_'+wy+'.nc')
        hf.get_gridded_files({"dataset": "CW3E", "temporal_resolution": "daily", "start_time": start_date, "end_time": end_date_cw3e, 
                              "grid_point": [bounds[0], bounds[1]]}, variables=variables, filename_template="scratch/"+site_id+"_{wy}.nc")
    
    ds = xr.open_dataset('scratch/'+site_id+'_'+wy+'.nc', engine='netcdf4', drop_variables=['y', 'x','latitude','longitude', 'Temp_min', 'Temp_max'])
    df = ds.to_dataframe()

    # if(df.size != 365):
    #     os.remove('scratch/'+site_id+'_'+wy+'.nc')
    #     hf.get_gridded_files({"dataset": "CW3E", "temporal_resolution": "daily", "start_time": start_date, "end_time": end_date_cw3e, 
    #                           "grid_point": [bounds[0], bounds[1]]}, variables=variables, filename_template="scratch/"+site_id+"_{wy}.nc")
    #     ds = xr.open_dataset('scratch/'+site_id+'_'+wy+'.nc', engine='netcdf4', drop_variables=['y', 'x','latitude','longitude', 'Temp_min', 'Temp_max'])
    #     df = ds.to_dataframe()
        
    
    met_data = pd.DataFrame({"DSWR":df['DSWR'].values, "DLWR":df['DLWR'].values, "precip":df['APCP'].values, "temp":df['Temp_mean'].values, 
                             "wind (E)":df['UGRD'].values, "wind (N)":df['VGRD'].values,"pressure":df['Press_atmos'].values, "q":df['SPFH'].values})
    print('loaded CW3E data')

    ## GET TOPOGRAPHICAL DATA ##
    variables = ["veg_type_IGBP", "slope_x", "slope_y"]
    if (not os.path.exists('scratch/'+site_id+'_'+wy+'_static.nc')):
        hf.get_gridded_files({"dataset": "conus2_domain", "grid":"conus2", "start_time": start_date, "end_time": end_date_cw3e, "grid_point": [bounds[0],
                                                                                                                                          bounds[1]]},
                             variables=variables, filename_template="scratch/"+site_id+"_{wy}_static.nc")
    
    ds = xr.open_dataset('scratch/'+site_id+'_'+wy+'_static.nc', engine='netcdf4', drop_variables=['y', 'x','latitude','longitude'])
    df = ds.to_dataframe()
    
    top_data = pd.DataFrame({"land_cover":df['vegetation_type'].values, "slope_x":df['slope_x'].values, "slope_y":df['slope_y'].values}, index=[0])
    print('loaded topographical data')
    
    # PROCESS
    # precip: fill nan values with zero
    precip['year'] = pd.DatetimeIndex(precip['date']).year
    precip = precip.set_index('date')
    precip['precip'] = precip['precip'].fillna(0)
    cols = precip.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    precip = precip[cols]
    
    # temp: fill nan values with linear interpolation
    temp = temp.set_index('date')
    temp['temp'] = temp['temp'].interpolate(method='linear', limit_direction='both')
    
    # swe: fill nan values with linear interpolation
    tot_swe['year'] = pd.DatetimeIndex(tot_swe['date']).year
    tot_swe = tot_swe.set_index('date')
    tot_swe['swe'] = tot_swe['swe'].interpolate(method='linear', limit_direction='both')
    
    # count years
    years = np.unique(tot_swe['year'])
    years = years[~np.isnan(years)]
    num_values = len(years)
    
    # combine all datasets and add SNOTEL metadata characteristitcs
    tot_non_swe = precip.join(temp)
    tot_non_swe['elevation'] = metadata_df['usda_elevation'][0]
    tot_non_swe['latitude'] = metadata_df['latitude'][0]
    tot_non_swe['longitude'] = metadata_df['longitude'][0]

    # add CW3E forcing characteristics
    tot_non_swe['DSWR'] = met_data['DSWR'].to_numpy().reshape(-1,1)
    tot_non_swe['DLWR'] = met_data['DLWR'].to_numpy().reshape(-1,1)
    tot_non_swe['wind (E)'] = met_data['wind (E)'].to_numpy().reshape(-1,1)
    tot_non_swe['wind (N)'] = met_data['wind (N)'].to_numpy().reshape(-1,1)
    tot_non_swe['pressure'] = met_data['pressure'].to_numpy().reshape(-1,1)
    tot_non_swe['q'] = met_data['q'].to_numpy().reshape(-1,1)

    # add topographic forcing characteristics
    tot_non_swe['land_cover'] = top_data['land_cover'][0]
    tot_non_swe['slope_x'] = top_data['slope_x'][0]
    tot_non_swe['slope_y'] = top_data['slope_y'][0]

    # last check for nan values
    tot_non_swe = tot_non_swe.fillna(0)
    
    # change years to water years - assume dates of swe/non swe are the same
    leap=[]
    for i in range(0,len(tot_swe)-1):
        d_non_swe = pd.to_datetime(tot_non_swe.index[i])
        d_swe = pd.to_datetime(tot_swe.index[i])
        if(d_non_swe.month >= 10):
            tot_non_swe.loc[tot_non_swe.index[i],'year'] += 1
        if(d_swe.month >= 10):
            tot_swe.loc[tot_swe.index[i],'year'] += 1
        if((d_non_swe.month==2) and (d_non_swe.day==29)):
            leap.append(tot_swe.index[i])
            
    # get rid of leap days
    tot_swe = tot_swe.drop(leap, axis=0)
    tot_non_swe = tot_non_swe.drop(leap, axis=0)

    # add site_id, reindex to remove date
    tot_non_swe['site_id'] = site_id
    tot_swe['site_id'] = site_id
    tot_non_swe = tot_non_swe.reset_index().drop(columns='date')
    tot_swe = tot_swe.reset_index().drop(columns='date')

    return tot_swe, tot_non_swe

## GET SINGLE COLUMN DATA FOR PARFLOW ##
# returns SNOTEL and CW3E data for point location at hourly time scale 
# assume start_date and end_date are water year adjusted
def get_sc_data_pf(site_id, start_date, end_date):
    # GET SNOTEL DATA 
    site_df_precip = hf.get_point_data(dataset='snotel', variable='precipitation', temporal_resolution='daily', aggregation='sum', 
                                   date_start=start_date, date_end=end_date, site_ids=[site_id])
    site_df_temp = hf.get_point_data(dataset='snotel', variable='air_temp', temporal_resolution='daily', aggregation='mean', 
                                 date_start=start_date, date_end=end_date, site_ids=[site_id])
    metadata_df = hf.get_point_metadata(dataset='snotel', variable='swe', temporal_resolution='daily', aggregation='sod', 
                                             date_start=start_date, date_end=end_date, site_ids=[site_id])
    print('loaded SNOTEL data')
        
    # PARAMETERS FOR CW3E DATA
    precip = site_df_precip.set_axis(['date','precip'], axis='columns')
    temp = site_df_temp.set_axis(['date','temp'], axis='columns')

    # adjust end date for CW3E
    end_date_cw3e = str(pd.to_datetime(end_date) + DateOffset(days=1))
    wy = str(pd.to_datetime(end_date).year)

    # create path
    path =  os.path.join('output', site_id+str(wy))
    if not os.path.exists(path):
       os.makedirs(path)
    
    lat = metadata_df['latitude'][0]               	
    lon = metadata_df['longitude'][0]              
    bounds = st.latlon_to_ij([[lat, lon],[lat, lon]],"conus2")
    
    # GET CW3E DATA
    variables = ["precipitation", "downward_longwave", "downward_shortwave", "specific_humidity", "air_temp", "east_windspeed", "north_windspeed", 
                 "atmospheric_pressure"]
    if (not os.path.exists('scratch/'+site_id+'_'+wy+'.nc')):
        hf.get_gridded_files({"dataset": "CW3E", "temporal_resolution": "hourly", "start_time": start_date, "end_time": end_date_cw3e, 
                              "grid_point": [bounds[0], bounds[1]]}, variables=variables, filename_template="scratch/"+site_id+"_{wy}.nc")
    
    ds = xr.open_dataset('scratch/'+site_id+'_'+wy+'.nc', engine='netcdf4', drop_variables=['y', 'x','latitude','longitude', 'Temp_min', 'Temp_max'])
    df = ds.to_dataframe()
        
    
    met_data = pd.DataFrame({"DSWR":df['DSWR'].values, "DLWR":df['DLWR'].values, "precip":df['APCP'].values, "temp":df['Temp_mean'].values, 
                             "wind (E)":df['UGRD'].values, "wind (N)":df['VGRD'].values,"pressure":df['Press_atmos'].values, "q":df['SPFH'].values})
    
    print('loaded CW3E data')

    # PROCESS
    # precip: fill nan values with zero
    precip = precip.set_index('date')
    precip['precip'] = precip['precip'].fillna(0)
    
    # temp: fill nan values with linear interpolation
    temp = temp.set_index('date')
    temp['temp'] = temp['temp'].interpolate(method='linear', limit_direction='both')

    # BIAS CORRECTION
    # based on SNOTEL data
    met = met_data['temp']-273.15
    sno = temp['temp']
    temp_mean = (sno-met).mean()

    met = met_data['precip']
    sno = precip['precip']
    precip_mean = (sno-met).mean()

    met_data['temp'] = met_data['temp']+temp_mean
    met_data['precip'] = met_data['precip']+(precip_mean/86400)

    met_data.to_csv(os.path.join(path, site_id + '_' + str(wy)+'_forcing.txt'),sep=' ',header=None, index=False, index_label=False)

    return met_data

## GET NORMALIZATION FUNCTIONS ##
# return list of normalization functions, SWE as the first one
def create_normalization(tot_swe, tot_non_swe):
    l_normalize = []
    # for swe variables - define scaling object for swe so data can be inverted later
    scaler_swe = MaxAbsScaler().fit(tot_swe[['swe']])
    l_normalize.append(scaler_swe)

    # for non-swe variables
    # ignore 0 (the year) and last (the site_id)
    for i in range(1,len(tot_non_swe.columns)-1):
        variable = tot_non_swe.columns[i]
        scaler = MaxAbsScaler().fit(tot_non_swe[[variable]])
        l_normalize.append(scaler)

    return l_normalize

## GENERATE DATA FOR TRAINING/TESTING ##
# return 3D array of tensors, with data organized by year
def create_dataset(tot_swe, tot_non_swe, l_normalize):
    # NORMALIZE DATA
    # for non-swe variables
    for i in range(1,len(tot_non_swe.columns)-1):
        variable = tot_non_swe.columns[i]
        scaler = l_normalize[i]
        tot_non_swe[variable] = scaler.transform(tot_non_swe[[variable]])

    # for swe variables
    scaler_swe = l_normalize[0]
    tot_swe['swe'] = scaler_swe.transform(tot_swe[['swe']])
    
    # CREATE DATASETS
    # add in processing by site_id
    sites, ind = np.unique(tot_swe['site_id'], return_index=True)
    sites = sites[np.argsort(ind)]
    
    l_swe = []
    l_non_swe = []
    l_sites = []
    l_years = []
    for j in range(0, len(sites)):
        site_swe = tot_swe.loc[tot_swe['site_id'] == sites[j]]
        site_swe = site_swe.loc[:, site_swe.columns != 'site_id']
        site_non_swe = tot_non_swe.loc[tot_non_swe['site_id'] == sites[j]]
        site_non_swe = site_non_swe.loc[:, site_non_swe.columns != 'site_id']
        years, ind = np.unique(site_swe['year'], return_index=True)
        years = years[np.argsort(ind)]
        years = years[~np.isnan(years)]
        
        for i in range(0, len(years)):
            year_swe = site_swe.loc[site_swe['year'] == years[i]]
            temp_swe = year_swe.loc[:, year_swe.columns != 'year'].to_numpy()
            year_non_swe = site_non_swe.loc[site_non_swe['year'] == years[i]]
            temp_non_swe = year_non_swe.loc[:, year_non_swe.columns != 'year'].to_numpy()
            
            # get rid of years without 365 days of data
            # consider making a more nuanced way to filter data?
            if((temp_swe.size == (365*temp_swe.shape[1])) and (temp_non_swe.size == (365*temp_non_swe.shape[1]))):
                l_swe.append(temp_swe)
                l_non_swe.append(temp_non_swe)
                l_sites.append(sites[j])
                l_years.append(years[i])
        
    # create arrays with all years
    full_swe = np.stack(l_swe).astype(np.float32)
    full_non_swe = np.stack(l_non_swe).astype(np.float32)
    full_sites = np.stack(l_sites)
    full_years = np.stack(l_years).astype(np.int_)
    num_years = len(full_years)

    full_swe_tensors = torch.from_numpy(full_swe)
    full_non_swe_tensors = torch.from_numpy(full_non_swe)

    #return full_swe, full_non_swe, full_sites, full_years, num_years,
    return full_swe_tensors, full_non_swe_tensors, full_sites, full_years

## GET RANDOM DATA ##
# given a start and end point, randomly select sample_size years
def get_years_random(start_date, end_date, site_id, sample_size):
    # adjust for water year naming convention
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    start_year = start.year + 1
    end_year = end.year + 1
    
    years = np.arange(start_year, end_year).tolist()

    #rn.seed(8)
    sample = rn.sample(years, sample_size)
    return sample

## GET DATA BINNED BY PRECIPITATION ##
# sample size is number of samples in each bin
# split into below avg, avg, above avg swe
def get_years_precip(start_date, end_date, site_id, sample_size):
    # GET DATA
    swe = hf.get_point_data(dataset='snotel', variable='swe', temporal_resolution='daily', aggregation='sod', 
                                             date_start=start_date, date_end=end_date, site_ids=[site_id])
    swe.columns = ['date','swe']
    swe['year'] = pd.DatetimeIndex(swe['date']).year
    swe = swe.set_index('date')
    
    # change years to water years, get rid of leap years
    leap=[]
    for i in range(0,len(swe)-1):
        d_swe = pd.to_datetime(swe.index[i])
        if(d_swe.month >= 10):
            swe.loc[swe.index[i],'year'] += 1
        if((d_swe.month==2) and (d_swe.day==29)):
            leap.append(swe.index[i])
    swe = swe.drop(leap, axis=0)
    swe = swe.reset_index().drop(columns='date')
    
    years = np.unique(swe['year'])

    # BIN SWE
    peak_values = np.column_stack((years, np.zeros(len(years))))
    for i in range(0,len(years)):
        yr = years[i]
        df_swe = swe.loc[swe['year'] == yr]
        peak_values[i,1] = max(df_swe['swe'])
    peak_values = pd.DataFrame(peak_values, columns = ['year','peak swe'])
    peak_values['bins'] = pd.qcut(peak_values['peak swe'], q=3, labels=['below avg', 'avg', 'above avg'])

    # RANDOMLY SELECT 
    #rn.seed(6)
    abv_avg_i = rn.sample(peak_values.loc[peak_values['bins'] == 'above avg'].index.to_list(), sample_size)
    avg_i = rn.sample(peak_values.loc[peak_values['bins'] == 'avg'].index.to_list(), sample_size)
    bel_avg_i = rn.sample(peak_values.loc[peak_values['bins'] == 'below avg'].index.to_list(), sample_size)
    
    sample = np.zeros(int(sample_size*3))
    for i in range(0, sample_size):
        sample[0+i] = years[abv_avg_i[i]]
        sample[sample_size+i] = years[avg_i[i]]
        sample[int(sample_size*2)+i] = years[bel_avg_i[i]]

    return sample


## NASH-SUTCLIFFE EFFICIENCY ##
def nse(actual, predictions):
    return (1-(np.sum((actual-predictions)**2)/np.sum((actual-np.mean(actual))**2)))


## ANALYZE DATA ##
# for given LSTM model run and supplementary data
def analyze_results(model, metadata, test_swe, test_non_swe, scaler_swe, gen_plot):
    
    num_years = len(metadata)
    statistics = pd.DataFrame(0.0, index=np.arange(num_years), columns=['rmse', 'normal rmse', 'nse', 'r2', 'spearman_rho', 'delta peak', 
                                                                        'normal delta peak', 'delta days'])
    l_features = []
    
    ig = IntegratedGradients(model)
    
    if gen_plot: 
        plt.figure(figsize=(32,24))

    for i in range(0, num_years):
        site_id = metadata['site_id'][i]
        year = metadata['site_id'][i]
        test_swe_tensors = test_swe[i]
        test_non_swe_tensors = test_non_swe[i].to(DEVICE)
        test_non_swe_tensors = torch.reshape(test_non_swe_tensors, (test_non_swe_tensors.shape[0], 1, test_non_swe_tensors.shape[1]))
    
        # predict 
        swe_pred = model(test_non_swe_tensors)
    
        # calculate feature attribution
        attr, delta = ig.attribute(test_non_swe_tensors.requires_grad_(), return_convergence_delta=True)
        attr = attr.cpu().detach().numpy()
        l_features.append(np.mean(attr, axis=0)[0])
    
        # inverse transform to produce swe values
        swe_pred = scaler_swe.inverse_transform(swe_pred.cpu().detach().numpy().reshape(-1,1))
        swe_actual = scaler_swe.inverse_transform(test_swe_tensors.detach().numpy())
    
        # peak swe
        peak_lstm = max(swe_pred)
        peak_obs = max(swe_actual)
        peak = (peak_lstm + peak_obs)/2
    
        # calculate RMSE
        mse = mean_squared_error(swe_actual, swe_pred)
        rmse = np.sqrt(mse)
        statistics.loc[i, 'rmse'] = rmse
        statistics.loc[i, 'normal rmse'] = rmse / peak
    
        # calculate NSE
        nash_sut = nse(swe_actual, swe_pred)
        statistics.loc[i, 'nse'] = nash_sut
    
        # calculate r2
        r_2 = r2_score(swe_actual, swe_pred)
        statistics.loc[i, 'r2'] = r_2
    
        # calculate Spearman's rho
        spearman_rho = stats.spearmanr(swe_actual, swe_pred)
        statistics.loc[i, 'spearman_rho'] = spearman_rho[0]
    
        # calculate delta peak SWE
        statistics.loc[i, 'delta peak'] = peak_lstm - peak_obs
        statistics.loc[i, 'normal delta peak'] = (peak_lstm - peak_obs) / peak
    
        # calculate first snow free day
        # obs/clm: swe == 0
        # pred: swe < 0
        arr_lstm = np.where(swe_pred < 0)[0]
        arr_obs = np.where(swe_actual == 0)[0]
        melt_lstm = np.where(arr_lstm > 100)[0]
        melt_obs = np.where(arr_obs > 100)[0]
        try:
            statistics.loc[i, 'delta days'] = arr_lstm[melt_lstm[0]] - arr_obs[melt_obs[0]]
        except:
            statistics.loc[i, 'delta days'] = 365 - arr_obs[melt_obs[0]]
    
        # plot first 36 years at 9 sites if boolean is true
        if gen_plot and (i < 36):
            plt.subplot(6, 6, i+1)
            # blue is actual, red is predicted
            plt.plot(swe_pred, label='predicted swe', c='red')
            plt.plot(swe_actual, label='actual swe', c='blue')
            plt.title(site_id + ': '+ str(year))
            #plt.title(f'{test_years['years'][y]:.0f}: RMSE: {rmse:.2f}') 
            plt.xlabel('days in WY')
            plt.ylabel('SWE [mm]')

    if gen_plot:
        plt.tight_layout()
    full = np.stack(l_features).astype(np.float32)

    return statistics, full


## ANALYZE DATA ##
# for given LSTM model run in comparison with ParFlow-CLM and UA SWE model
def analyze_results_allmodels(model, metadata, test_swe, test_non_swe, scaler_swe, gen_plot):

    values = xr.open_dataset('/home/mcburns/UA_SWE_data/4km_SWE_Depth_WY2000_v01.nc')
    lat_values = values['lat'].values
    lon_values = values['lon'].values
    
    num_years = len(metadata)
    
    if gen_plot: 
        plt.figure(figsize=(32,24))

    for i in range(0, num_years):
        site_id = metadata['site_id'][i]
        year = metadata['year'][i]
        lat = metadata['latitude'][i]
        lon = metadata['longitude'][i]
        test_swe_tensors = test_swe[i]
        test_non_swe_tensors = test_non_swe[i].to(DEVICE)
        test_non_swe_tensors = torch.reshape(test_non_swe_tensors, (test_non_swe_tensors.shape[0], 1, test_non_swe_tensors.shape[1]))

        # GET LSTM PREDICTIONS
        # predict 
        swe_pred = model(test_non_swe_tensors)
        # inverse transform to produce swe values
        swe_pred = scaler_swe.inverse_transform(swe_pred.cpu().detach().numpy().reshape(-1,1))
        swe_actual = scaler_swe.inverse_transform(test_swe_tensors.detach().numpy())

        # GET UA SWE PREDICTIONS
        for j in np.arange(1, len(lat_values)):
            l_current = lat_values[j]
            if (lat <= l_current): #and (lat > l_last):
                lat_coord = lat_values[j-1]      #i-1
                break
        for j in np.arange(1, len(lon_values)):
            l_last = lon_values[j-1]
            l_current = lon_values[j]
            if (lat >= l_last) and (lon < l_current):
                lon_coord = lon_values[j-1]
                break
        values = xr.open_dataset('/home/mcburns/UA_SWE_data/4km_SWE_Depth_WY'+str(year)+'_v01.nc')
        swe_ua = values.sel(lat=lat_coord, lon=lon_coord)['SWE'].values

        # GET PF CLM PREDICTIONS
        clm_output = pd.read_csv(os.path.join('/home/mcburns/pfclm/output/'+site_id+'/'+str(year), 'clm_output.txt'),sep=' ',header=None, index_col=None)
        clm_output.columns = ['LH [W/m2]', 'T [mm/s]', 'Ebs [mm/s]', 'Qflux infil [mm/s]', 'qflx_evap_tot [mm/s]', 'qflx_evap_grnd [mm/s]',
                              'SWE [mm]', 'Tgrnd [K]']
        # adjust PFCLM output to daily resolution
        swe_clm = np.zeros(365)
        for k in range(0,365):
            i_hr = k*24
            avg = np.mean(clm_output['SWE [mm]'][i_hr:i_hr+24])
            swe_clm[k] = avg
        swe_clm = pd.DataFrame(swe_clm,columns=['swe'])
        swe_clm = swe_clm.interpolate(method='linear', limit_direction='both')

        # SWE ACTUAL 2.0
        swe_actual_test = pd.read_csv(os.path.join('/home/mcburns/pfclm/output/'+site_id+'/'+str(year), site_id+'_'+str(year)+'_swe.txt'), 
                                      sep=' ',header=None,index_col=False)
        swe_actual_test.columns = ['date', 'swe']
    
        # plot first 36 years at 9 sites if boolean is true
        if gen_plot and (i < 36):
            plt.subplot(6, 6, i+1)
            # blue is actual, red is predicted
            plt.plot(swe_pred, label='LSTM', c='darkred')
            plt.plot(swe_actual, label='SNOTEL', c='#203864')
            plt.plot(swe_clm, label='CLM', c='green')
            plt.plot(swe_ua, label='UA SWE', c='darkmagenta')
            plt.title(site_id + ': '+ str(year))
            #plt.title(f'{test_years['years'][y]:.0f}: RMSE: {rmse:.2f}') 
            plt.xlabel('Days in WY')
            plt.ylabel('SWE [mm]')
            plt.legend()

    if gen_plot:
        plt.tight_layout()

    return 
    

## PRINT AND VISUALIZE FEATURE IMPORTANCES ##
def visualize_importances(feature_names, importances, title="Average Feature Importances", plot=True, axis_title="Features"):
    print(title)
    for i in range(len(feature_names)):
        print(feature_names[i], ": ", '%.3f'%(importances[i]))
    x_pos = (np.arange(len(feature_names)))
    if plot:
        plt.figure(figsize=(12,6))
        plt.bar(x_pos, importances, align='center')
        plt.xticks(x_pos, feature_names, wrap=True)
        plt.xlabel(axis_title)
        plt.title(title)


## GET METRICS FOR  MODEL RUNS ##
# returns metrics for model runs given two lists of model names
def get_model_metrics(models):
    total_statistics = pd.DataFrame(columns=['run name','rmse', 'normal rmse', 'nse', 'r2', 'spearman_rho', 'delta peak', 'normal delta peak', 'delta days'])
    for run in models:
        statistics = pd.read_csv('output/'+run+'_statistics.txt',sep=' ',header=None)
        statistics.columns = ['rmse', 'normal rmse', 'nse', 'r2', 'spearman_rho', 'delta peak', 'normal delta peak', 'delta days']
        total_statistics.loc[len(total_statistics)] = [run, np.mean(statistics['rmse']), np.mean(statistics['normal rmse']), np.mean(statistics['nse']), 
                                                       np.mean(statistics['r2']),np.mean(statistics['spearman_rho']), 
                                                       np.mean(statistics['delta peak']), np.mean(statistics['normal delta peak']), 
                                                       np.mean(statistics['delta days'])]
    total_statistics['normal delta days'] = total_statistics['delta days']/365
    
    return total_statistics
    
    
