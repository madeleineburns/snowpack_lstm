## FUNCTIONS FOR RUNNING PFCLM MODEL ##

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
import torch
from glob import glob

# for data
import subsettools as st
import hf_hydrodata as hf
from pandas.tseries.offsets import DateOffset

from parflow import Run
import shutil
import parflow as pf
from parflow.tools.fs import mkdir, cp, chdir, get_absolute_path, rm, exists
import parflow.tools.hydrology as hydro
import netCDF4

from contextlib import redirect_stdout
trap = io.StringIO()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


## RUN PARFLOW ##
def run_pf(site_id, start_date, end_date):
    # directories and run name
    os.environ["PARFLOW_DIR"] = "/home/SHARED/software/parflow/3.10.0"
    
    wy = str(pd.to_datetime(end_date).year)
    base = '/home/mcburns/pfclm/output/'+site_id+'/'+wy
    #print(base)
    PFCLM_SC = Run("pfclm")
    stopt = 8760                ## run for 365 days
    
    # File input version number
    PFCLM_SC.FileVersion = 4
    # Process Topology
    PFCLM_SC.Process.Topology.P = 1
    PFCLM_SC.Process.Topology.Q = 1
    PFCLM_SC.Process.Topology.R = 1
    # Computational Grid
    PFCLM_SC.ComputationalGrid.Lower.X = 0.0
    PFCLM_SC.ComputationalGrid.Lower.Y = 0.0
    PFCLM_SC.ComputationalGrid.Lower.Z = 0.0
    PFCLM_SC.ComputationalGrid.DX      = 2.0
    PFCLM_SC.ComputationalGrid.DY      = 2.0
    PFCLM_SC.ComputationalGrid.DZ      = 1.0
    PFCLM_SC.ComputationalGrid.NX      = 1
    PFCLM_SC.ComputationalGrid.NY      = 1
    PFCLM_SC.ComputationalGrid.NZ      = 10
    # The Names of the GeomInputs
    PFCLM_SC.GeomInput.Names = 'domain_input'
    # Domain Geometry Input
    PFCLM_SC.GeomInput.domain_input.InputType = 'Box'
    PFCLM_SC.GeomInput.domain_input.GeomName  = 'domain'
    # Domain Geometry
    PFCLM_SC.Geom.domain.Lower.X = 0.0
    PFCLM_SC.Geom.domain.Lower.Y = 0.0
    PFCLM_SC.Geom.domain.Lower.Z = 0.0
    PFCLM_SC.Geom.domain.Upper.X = 2.0
    PFCLM_SC.Geom.domain.Upper.Y = 2.0
    PFCLM_SC.Geom.domain.Upper.Z = 10.0
    PFCLM_SC.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'
    # variable dz assignments
    PFCLM_SC.Solver.Nonlinear.VariableDz = True
    PFCLM_SC.dzScale.GeomNames           = 'domain'
    PFCLM_SC.dzScale.Type                = 'nzList'
    PFCLM_SC.dzScale.nzListNumber        = 10
    # cells start at the bottom (0) and moves up to the top
    # domain is 8 m thick, root zone is down to 4 cells 
    # so the root zone is 2 m thick
    PFCLM_SC.Cell._0.dzScale.Value  = 1.0   
    PFCLM_SC.Cell._1.dzScale.Value  = 1.0   
    PFCLM_SC.Cell._2.dzScale.Value  = 1.0   
    PFCLM_SC.Cell._3.dzScale.Value  = 1.0
    PFCLM_SC.Cell._4.dzScale.Value  = 1.0
    PFCLM_SC.Cell._5.dzScale.Value  = 1.0
    PFCLM_SC.Cell._6.dzScale.Value  = 1.0
    PFCLM_SC.Cell._7.dzScale.Value  = 0.6   #0.6* 1.0 = 0.6  60 cm 3rd layer
    PFCLM_SC.Cell._8.dzScale.Value  = 0.3   #0.3* 1.0 = 0.3  30 cm 2nd layer
    PFCLM_SC.Cell._9.dzScale.Value  = 0.1   #0.1* 1.0 = 0.1  10 cm top layer
    # Perm
    PFCLM_SC.Geom.Perm.Names              = 'domain'
    PFCLM_SC.Geom.domain.Perm.Type        = 'Constant'
    PFCLM_SC.Geom.domain.Perm.Value       = 0.01465  # m/h 
    PFCLM_SC.Perm.TensorType              = 'TensorByGeom'
    PFCLM_SC.Geom.Perm.TensorByGeom.Names = 'domain'
    PFCLM_SC.Geom.domain.Perm.TensorValX  = 1.0
    PFCLM_SC.Geom.domain.Perm.TensorValY  = 1.0
    PFCLM_SC.Geom.domain.Perm.TensorValZ  = 1.0
    # Specific Storage
    PFCLM_SC.SpecificStorage.Type              = 'Constant'
    PFCLM_SC.SpecificStorage.GeomNames         = 'domain'
    PFCLM_SC.Geom.domain.SpecificStorage.Value = 1.0e-4
    # Phases
    PFCLM_SC.Phase.Names = 'water'
    PFCLM_SC.Phase.water.Density.Type     = 'Constant'
    PFCLM_SC.Phase.water.Density.Value    = 1.0
    PFCLM_SC.Phase.water.Viscosity.Type   = 'Constant'
    PFCLM_SC.Phase.water.Viscosity.Value  = 1.0
    # Contaminants
    PFCLM_SC.Contaminants.Names = ''
    # Gravity
    PFCLM_SC.Gravity = 1.0
    # Setup timing info
    PFCLM_SC.TimingInfo.BaseUnit     = 1.0
    PFCLM_SC.TimingInfo.StartCount   = 0
    PFCLM_SC.TimingInfo.StartTime    = 0.0
    PFCLM_SC.TimingInfo.StopTime     = stopt
    PFCLM_SC.TimingInfo.DumpInterval = 1.0
    PFCLM_SC.TimeStep.Type           = 'Constant'
    PFCLM_SC.TimeStep.Value          = 1.0
    # Porosity
    PFCLM_SC.Geom.Porosity.GeomNames    = 'domain'
    PFCLM_SC.Geom.domain.Porosity.Type  = 'Constant'
    PFCLM_SC.Geom.domain.Porosity.Value = 0.3
    # Domain
    PFCLM_SC.Domain.GeomName = 'domain'
    # Mobility
    PFCLM_SC.Phase.water.Mobility.Type  = 'Constant'
    PFCLM_SC.Phase.water.Mobility.Value = 1.0
    # Relative Permeability
    PFCLM_SC.Phase.RelPerm.Type        = 'VanGenuchten'
    PFCLM_SC.Phase.RelPerm.GeomNames   = 'domain'
    PFCLM_SC.Geom.domain.RelPerm.Alpha = 2.0
    PFCLM_SC.Geom.domain.RelPerm.N     = 3.0
    # Saturation
    PFCLM_SC.Phase.Saturation.Type        = 'VanGenuchten'
    PFCLM_SC.Phase.Saturation.GeomNames   = 'domain'
    PFCLM_SC.Geom.domain.Saturation.Alpha = 2.0
    PFCLM_SC.Geom.domain.Saturation.N     = 3.0
    PFCLM_SC.Geom.domain.Saturation.SRes  = 0.2
    PFCLM_SC.Geom.domain.Saturation.SSat  = 1.0
    # Wells
    PFCLM_SC.Wells.Names = ''
    
    # Time Cycles
    PFCLM_SC.Cycle.Names = 'constant'
    PFCLM_SC.Cycle.constant.Names = 'alltime'
    PFCLM_SC.Cycle.constant.alltime.Length = 1
    PFCLM_SC.Cycle.constant.Repeat = -1
    # Boundary Conditions: Pressure
    PFCLM_SC.BCPressure.PatchNames = 'x_lower x_upper y_lower y_upper z_lower z_upper'
    PFCLM_SC.Patch.x_lower.BCPressure.Type          = 'FluxConst'
    PFCLM_SC.Patch.x_lower.BCPressure.Cycle         = 'constant'
    PFCLM_SC.Patch.x_lower.BCPressure.alltime.Value = 0.0
    PFCLM_SC.Patch.y_lower.BCPressure.Type          = 'FluxConst'
    PFCLM_SC.Patch.y_lower.BCPressure.Cycle         = 'constant'
    PFCLM_SC.Patch.y_lower.BCPressure.alltime.Value = 0.0
    PFCLM_SC.Patch.z_lower.BCPressure.Type          = 'DirEquilRefPatch'
    PFCLM_SC.Patch.z_lower.BCPressure.RefGeom       = 'domain'
    PFCLM_SC.Patch.z_lower.BCPressure.RefPatch      = 'z_upper'
    PFCLM_SC.Patch.z_lower.BCPressure.Cycle         = 'constant'
    PFCLM_SC.Patch.z_lower.BCPressure.alltime.Value = -0.5 
    PFCLM_SC.Patch.x_upper.BCPressure.Type          = 'FluxConst'
    PFCLM_SC.Patch.x_upper.BCPressure.Cycle         = 'constant'
    PFCLM_SC.Patch.x_upper.BCPressure.alltime.Value = 0.0
    PFCLM_SC.Patch.y_upper.BCPressure.Type          = 'FluxConst'
    PFCLM_SC.Patch.y_upper.BCPressure.Cycle         = 'constant'
    PFCLM_SC.Patch.y_upper.BCPressure.alltime.Value = 0.0
    PFCLM_SC.Patch.z_upper.BCPressure.Type          = 'OverlandFlow'
    PFCLM_SC.Patch.z_upper.BCPressure.Cycle         = 'constant'
    PFCLM_SC.Patch.z_upper.BCPressure.alltime.Value = 0.0
    # Topo slopes in x-direction
    PFCLM_SC.TopoSlopesX.Type              = 'Constant'
    PFCLM_SC.TopoSlopesX.GeomNames         = 'domain'
    PFCLM_SC.TopoSlopesX.Geom.domain.Value = 0.1  #slope in X-direction to allow ponded water to run off
    # Topo slopes in y-direction
    PFCLM_SC.TopoSlopesY.Type              = 'Constant'
    PFCLM_SC.TopoSlopesY.GeomNames         = 'domain'
    PFCLM_SC.TopoSlopesY.Geom.domain.Value = 0.0
    # Mannings coefficient
    PFCLM_SC.Mannings.Type               = 'Constant'
    PFCLM_SC.Mannings.GeomNames          = 'domain'
    PFCLM_SC.Mannings.Geom.domain.Value  = 2.e-6
    # Phase sources:
    PFCLM_SC.PhaseSources.water.Type              = 'Constant'
    PFCLM_SC.PhaseSources.water.GeomNames         = 'domain'
    PFCLM_SC.PhaseSources.water.Geom.domain.Value = 0.0
    # Exact solution specification for error calculations
    PFCLM_SC.KnownSolution = 'NoKnownSolution'
    
    # Set solver parameters
    PFCLM_SC.Solver         = 'Richards'
    PFCLM_SC.Solver.MaxIter = 15000
    PFCLM_SC.Solver.Nonlinear.MaxIter           = 100
    PFCLM_SC.Solver.Nonlinear.ResidualTol       = 1e-5
    PFCLM_SC.Solver.Nonlinear.EtaChoice         = 'Walker1'
    PFCLM_SC.Solver.Nonlinear.EtaValue          = 0.01
    PFCLM_SC.Solver.Nonlinear.UseJacobian       = False
    PFCLM_SC.Solver.Nonlinear.DerivativeEpsilon = 1e-12
    PFCLM_SC.Solver.Nonlinear.StepTol           = 1e-30
    PFCLM_SC.Solver.Nonlinear.Globalization     = 'LineSearch'
    PFCLM_SC.Solver.Linear.KrylovDimension      = 100
    PFCLM_SC.Solver.Linear.MaxRestarts          = 5
    PFCLM_SC.Solver.Linear.Preconditioner       = 'PFMG'
    PFCLM_SC.Solver.PrintSubsurf                = False
    PFCLM_SC.Solver.Drop                        = 1E-20
    PFCLM_SC.Solver.AbsTol                      = 1E-9
    
    #Writing output options for ParFlow
    write_pfb = True  #only PFB output for water balance example
    #  PFB  no SILO
    PFCLM_SC.Solver.PrintSubsurfData         = False
    PFCLM_SC.Solver.PrintPressure            = False
    PFCLM_SC.Solver.PrintSaturation          = False
    PFCLM_SC.Solver.PrintCLM                 = write_pfb
    PFCLM_SC.Solver.PrintMask                = False
    PFCLM_SC.Solver.PrintSpecificStorage     = False
    PFCLM_SC.Solver.PrintEvapTrans           = False
    
    PFCLM_SC.Solver.WriteSiloMannings        = False
    PFCLM_SC.Solver.WriteSiloMask            = False
    PFCLM_SC.Solver.WriteSiloSlopes          = False
    PFCLM_SC.Solver.WriteSiloSaturation      = False
    
    #write output in NetCDF - CAN COMMENT OUT PRESSURE & SATURATION
    write_netcdf = False
    #PFCLM_SC.NetCDF.NumStepsPerFile          = 8760
    #PFCLM_SC.NetCDF.WritePressure            = write_netcdf  
    PFCLM_SC.NetCDF.WriteSubsurface          = False
    #PFCLM_SC.NetCDF.WriteSaturation          = write_netcdf
    PFCLM_SC.NetCDF.WriteCLM                 = write_netcdf
    #PFCLM_SC.NetCDF.CLMNumStepsPerFile       = 240
    
    # LSM / CLM options - set LSM options to CLM
    PFCLM_SC.Solver.LSM              = 'CLM'
    # specify type of forcing, file name and location
    PFCLM_SC.Solver.CLM.MetForcing   = '1D'
    PFCLM_SC.Solver.CLM.MetFileName = site_id + '_' + str(wy)+'_forcing.txt'
    PFCLM_SC.Solver.CLM.MetFilePath  = '.'
    
    # Set CLM Plant Water Use Parameters
    PFCLM_SC.Solver.CLM.EvapBeta       = 'Linear'
    PFCLM_SC.Solver.CLM.VegWaterStress = 'Saturation'
    PFCLM_SC.Solver.CLM.ResSat         = 0.25
    PFCLM_SC.Solver.CLM.WiltingPoint   = 0.25
    PFCLM_SC.Solver.CLM.FieldCapacity  = 1.0       
    PFCLM_SC.Solver.CLM.IrrigationType = 'none'
    PFCLM_SC.Solver.CLM.RootZoneNZ     =  3   # layer used for seasonal Temp for LAI
    PFCLM_SC.Solver.CLM.SoiLayer       =  4   # root zone thickness, see above
    
    #PFCLM_SC.Solver.CLM.UseSlopeAspect = True
    
    #Writing output options for CLM
    #  no SILO, no native CLM logs
    PFCLM_SC.Solver.PrintLSMSink        = False
    PFCLM_SC.Solver.CLM.CLMDumpInterval = 1
    PFCLM_SC.Solver.CLM.CLMFileDir      = base
    PFCLM_SC.Solver.CLM.BinaryOutDir    = False
    PFCLM_SC.Solver.CLM.IstepStart      = 1
    PFCLM_SC.Solver.WriteCLMBinary      = False
    PFCLM_SC.Solver.WriteSiloCLM        = False
    PFCLM_SC.Solver.CLM.WriteLogs       = False
    PFCLM_SC.Solver.CLM.WriteLastRST    = True
    PFCLM_SC.Solver.CLM.DailyRST        = False
    PFCLM_SC.Solver.CLM.SingleFile      = True
    

    # Initial conditions: water pressure
    PFCLM_SC.ICPressure.Type                 = 'HydroStaticPatch'
    PFCLM_SC.ICPressure.GeomNames            = 'domain'
    PFCLM_SC.Geom.domain.ICPressure.Value    = 2.00
    PFCLM_SC.Geom.domain.ICPressure.RefGeom  = 'domain'
    PFCLM_SC.Geom.domain.ICPressure.RefPatch = 'z_lower'
    
    # Run ParFlow prior to changes
    PFCLM_SC.run(working_directory=base)
    return
    

## WATER YEAR START ##
# given datetime date, return start of next water year
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


## GET SINGLE COLUMN DATA FOR PARFLOW ##
# writes all PFCLM files to site_id and wy folders, returns nothing
# assume start_date and end_date are water year adjusted
def get_sc_data_pf(site_id, start_date, end_date):
    # CREATE PATH
    wy = str(pd.to_datetime(end_date).year)
    path =  os.path.join('output', site_id, str(wy))
    if not os.path.exists(path):
       os.makedirs(path)
        
    # GET SNOTEL DATA 
    site_df_swe = hf.get_point_data(dataset='snotel', variable='swe', temporal_resolution='daily', aggregation='sod', 
                                             date_start=start_date, date_end=end_date, site_ids=[site_id])
    site_df_precip = hf.get_point_data(dataset='snotel', variable='precipitation', temporal_resolution='daily', aggregation='sum', 
                                   date_start=start_date, date_end=end_date, site_ids=[site_id])
    site_df_temp = hf.get_point_data(dataset='snotel', variable='air_temp', temporal_resolution='daily', aggregation='mean', 
                                 date_start=start_date, date_end=end_date, site_ids=[site_id])
    metadata_df = hf.get_point_metadata(dataset='snotel', variable='swe', temporal_resolution='daily', aggregation='sod', 
                                             date_start=start_date, date_end=end_date, site_ids=[site_id])
        
    swe = site_df_swe.set_axis(['date','swe'], axis='columns')
    precip = site_df_precip.set_axis(['date','precip'], axis='columns')
    temp = site_df_temp.set_axis(['date','temp'], axis='columns')
    
    swe.to_csv(os.path.join(path, site_id + '_' + str(wy)+'_swe.txt'),sep=' ',header=None, index=False, index_label=False)
    print('loaded SNOTEL data')
    
    # PARAMETERS FOR CW3E DATA
    # adjust end date for CW3E
    end_date_cw3e = str(pd.to_datetime(end_date) + DateOffset(days=1))
    
    lat = metadata_df['latitude'][0]               	
    lon = metadata_df['longitude'][0]              
    bounds = st.latlon_to_ij([[lat, lon], [lat, lon]], "conus2")
    
    # GET CW3E DATA
    variables = ["precipitation", "downward_longwave", "downward_shortwave", "specific_humidity", "air_temp", "east_windspeed", "north_windspeed", 
                 "atmospheric_pressure"]
    if (not os.path.exists("output/"+site_id+"/"+str(wy)+"/hourly.nc")):
        hf.get_gridded_files({"dataset": "CW3E", "temporal_resolution": "hourly", "start_time": start_date, "end_time": end_date_cw3e, 
                              "grid_point": [bounds[0], bounds[1]]}, variables=variables, filename_template="output/"+site_id+"/"+str(wy)+"/hourly.nc")
    if (not os.path.exists("output/"+site_id+"/"+str(wy)+"/daily.nc")):
        hf.get_gridded_files({"dataset": "CW3E", "temporal_resolution": "daily", "start_time": start_date, "end_time": end_date_cw3e, 
                              "grid_point": [bounds[0], bounds[1]]}, variables=variables, filename_template="output/"+site_id+"/"+str(wy)+"/daily.nc")
    
    ds = xr.open_dataset('output/'+site_id+'/'+str(wy)+'/hourly.nc', engine='netcdf4', drop_variables=['y', 'x','latitude','longitude', 'Temp_min',
                                                                                                              'Temp_max'])
    df = ds.to_dataframe()
    met_data_hourly = pd.DataFrame({"DSWR":df['DSWR'].values, "DLWR":df['DLWR'].values, "precip":df['APCP'].values, "temp":df['Temp'].values, 
                             "wind (E)":df['UGRD'].values, "wind (N)":df['VGRD'].values,"pressure":df['Press'].values, "q":df['SPFH'].values})
    
    ds = xr.open_dataset("output/"+site_id+"/"+str(wy)+"/daily.nc", engine='netcdf4', drop_variables=['y', 'x','latitude','longitude', 'Temp_min',
                                                                                                             'Temp_max'])
    df = ds.to_dataframe()
    met_data_daily = pd.DataFrame({"DSWR":df['DSWR'].values, "DLWR":df['DLWR'].values, "precip":df['APCP'].values, "temp":df['Temp_mean'].values, 
                             "wind (E)":df['UGRD'].values, "wind (N)":df['VGRD'].values,"pressure":df['Press_atmos'].values, "q":df['SPFH'].values})
    
    print('loaded CW3E data')

    # PROCESS
    precip['precip'] = precip['precip'].fillna(0)
    temp['temp'] = temp['temp'].interpolate(method='linear', limit_direction='both')
    
    # BIAS CORRECTION
    # based on daily SNOTEL data, then apply to hourly
    test = temp['temp'] - (met_data_daily['temp'] - 273.15)
    temp_mean = test.mean()
    
    test = precip['precip'] - met_data_daily['precip']
    precip_mean = test.mean()
    
    met_data_hourly['temp'] = met_data_hourly['temp']+temp_mean
    met_data_hourly['precip'] = met_data_hourly['precip']+(precip_mean/86400)
    
    # SAVE DATA
    met_data_hourly.to_csv(os.path.join(path, site_id + '_' + str(wy)+'_forcing.txt'),sep=' ',header=None, index=False, index_label=False)
    
    # GET STATIC DATA
    static_filepaths = st.subset_static(bounds, dataset="conus2_domain", write_dir=path)
    clm_paths = st.config_clm(bounds, start=start_date, end=end_date, dataset="conus2_domain", write_dir=path)

    return


## GET SINGLE COLUMN METADATA FOR SITE
def get_metadata_pf(site_id, start_date, end_date):
    # preliminaries
    wy = str(pd.to_datetime(end_date).year)
    path =  os.path.join('output', site_id, str(wy))
    if not os.path.exists(path):
       os.makedirs(path)

    # get snotel metadata
    metadata_snotel = hf.get_point_metadata(dataset='snotel', variable='swe', temporal_resolution='daily', aggregation='sod', 
                                        date_start=start_date, date_end=end_date, site_ids=[site_id])
    
    lat = metadata_snotel['latitude'][0]               	
    lon = metadata_snotel['longitude'][0]              
    bounds = st.latlon_to_ij([[lat, lon],[lat, lon]],"conus2")
    
    # get site metadata
    variables = ["veg_type_IGBP", "slope_x", "slope_y"]
    if (not os.path.exists(path+'/'+site_id+'_'+wy+'_static.nc')):
        hf.get_gridded_files({"dataset": "conus2_domain", "grid":"conus2", "start_time": start_date, "end_time": end_date, "grid_point": [bounds[0],
                                                                                                                                          bounds[1]]},
                             variables=variables, filename_template=path+"/"+site_id+"_{wy}_static.nc")
    
    ds = xr.open_dataset(path+'/'+site_id+'_'+wy+'_static.nc', engine='netcdf4', drop_variables=['y', 'x','latitude','longitude'])
    metadata_top = ds.to_dataframe()
    
    metadata = pd.DataFrame({"elevation":metadata_snotel['usda_elevation'][0],"latitude":metadata_snotel['latitude'][0],
                             "longitude":metadata_snotel['longitude'][0], "land_cover":metadata_top['vegetation_type'].values,
                             "slope_x":metadata_top['slope_x'].values, "slope_y":metadata_top['slope_y'].values}, index=[0])
    return metadata


## GET RANDOM DATA ##
# given a start and end point, randomly select sample_size years
def get_years_random(start_date, end_date, site_id, sample_size):
    # adjust for water year naming convention
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    start_year = start.year + 1
    end_year = end.year + 1
    
    years = np.arange(start_year, end_year).tolist()

    rn.seed(8)
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
    rn.seed(6)
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

## PRODUCE RESULTS ##
def prod_swe(site_id, year):
    # read PFCLM output data
    try:
        clm_output = pd.read_csv(os.path.join('output/'+site_id+'/'+str(year), 'clm_output.txt'),sep=' ',header=None, index_col=None)
    except:
        return
    clm_output.columns = ['LH [W/m2]', 'T [mm/s]', 'Ebs [mm/s]', 'Qflux infil [mm/s]', 'qflx_evap_tot [mm/s]', 'qflx_evap_grnd [mm/s]','SWE [mm]', 'Tgrnd [K]']

    # adjust PFCLM output to daily resolution
    swe_clm = np.zeros(365)
    for i in range(0,365):
        i_hr = i*24
        avg = np.mean(clm_output['SWE [mm]'][i_hr:i_hr+24])
        swe_clm[i] = avg
    swe_clm = pd.DataFrame(swe_clm,columns=['swe'])
    swe_clm = swe_clm.interpolate(method='linear', limit_direction='both')

    # read actual SWE
    swe_actual = pd.read_csv(os.path.join('output/'+site_id+'/'+str(year), site_id+'_'+str(year)+'_swe.txt'), sep=' ',header=None,index_col=False)
    swe_actual.columns = ['date', 'swe']

    if len(swe_actual) == 366:
        swe_actual = swe_actual.drop(60).reset_index()

    swe_actual['swe'] =  swe_actual['swe'].interpolate(method='linear', limit_direction='both')

    return swe_clm, swe_actual


## ANALYZE DATA ##
def analyze_results(swe_model, swe_actual, site_id, year):
    statistics = pd.DataFrame(columns=['rmse', 'normal rmse', 'nse', 'r2', 'spearman_rho', 'delta peak', 'normal delta peak', 'delta days'])
    
    # peak swe
    peak_model = max(swe_model)
    peak_obs = max(swe_actual['swe'])
    peak = (peak_model + peak_obs)/2

    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(swe_actual['swe'], swe_model))

    # calculate NSE
    nash_sut = nse(swe_actual['swe'], swe_model)

    # calculate r2
    r_2 = r2_score(swe_actual['swe'], swe_model)

    # calculate Spearman's rho
    spearman_rho = stats.spearmanr(swe_actual['swe'], swe_model)

    # calculate delta peak SWE
    delta_peak = peak_model - peak_obs

    # calculate first snow free day
    # obs/clm: swe == 0
    # pred: swe < 0
    arr_lstm = np.where(swe_model == 0)[0]
    arr_obs = np.where(swe_actual['swe'] == 0)[0]
    melt_lstm = np.where(arr_lstm > 100)[0]
    melt_obs = np.where(arr_obs > 100)[0]
    try:
        delta_days = arr_lstm[melt_lstm[0]] - arr_obs[melt_obs[0]]
    except:
        delta_days = 365 - arr_obs[melt_obs[0]]

    statistics.loc[len(statistics)] = [rmse, rmse / peak, nash_sut, r_2, spearman_rho[0], delta_peak, delta_peak/peak, delta_days]
    
    return statistics


## GET METRICS FOR  MODEL RUNS ##
# returns metrics for model runs given two lists of model names
def get_model_metrics(models, spatial_distribution):
    total_statistics = pd.DataFrame(columns=['run name','rmse', 'normal rmse', 'nse', 'r2', 'spearman_rho', 'delta peak', 'normal delta peak', 'delta days'])
    for run in models:
        statistics = pd.read_csv('/home/mcburns/'+spatial_distribution+'_lstm/output/'+run+'_statistics.txt',sep=' ',header=None)
        statistics.columns = ['rmse', 'normal rmse', 'nse', 'r2', 'spearman_rho', 'delta peak', 'normal delta peak', 'delta days']
        total_statistics.loc[len(total_statistics)] = [run, np.mean(statistics['rmse']), np.mean(statistics['normal rmse']), np.mean(statistics['nse']), 
                                                       np.mean(statistics['r2']),np.mean(statistics['spearman_rho']), 
                                                       np.mean(np.abs(statistics['delta peak'])), np.mean(np.abs(statistics['normal delta peak'])), 
                                                       np.mean(statistics['delta days'])]
    
    return total_statistics
