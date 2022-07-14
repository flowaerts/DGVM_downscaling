# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 09:16:59 2022

@author: floweidinger
"""
################# IMPORTS #######################
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing as mp
from functools import partial   
from math import isnan

#imports from directory
from proxy_smoothing import proxy_smoother

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))


def ecoreadj_loop(ecoids, dgvm_eco, dgvm_adj_eco, land_eco, ecoregions_eco): 
    # area weighted mean of dgvm values in ecoregions masked with dgvm_adj (to get land & nonprod clip)
    
    if ecoids.size>0:
        eco_adjs = {}
        for i_eco in ecoids:
            dgvm_orig_sum = np.nansum(
                np.multiply(
                    dgvm_eco[np.where(ecoregions_eco==i_eco)][~np.isnan(dgvm_adj_eco[np.where(ecoregions_eco==i_eco)])], 
                    land_eco[np.where(ecoregions_eco==i_eco)][~np.isnan(dgvm_adj_eco[np.where(ecoregions_eco==i_eco)])]
                    ))
            
            sqkm_sum = np.nansum(land_eco[np.where(ecoregions_eco==i_eco)])
            dgvm_orig_mean =dgvm_orig_sum/sqkm_sum
            
            
            # area weighted mean of downscaled DGVM in ecoregions
            dgvm_down_sum = np.nansum(
               np.multiply(
                   dgvm_adj_eco[np.where(ecoregions_eco==i_eco)],
                   land_eco[np.where(ecoregions_eco==i_eco)]
                   )    
               )
            dgvm_down_mean = dgvm_down_sum/sqkm_sum
            
            
            # difference between downscaled ecoregion mean and original ecoregion mean
            diff_p = dgvm_orig_mean/dgvm_down_mean
            
            eco_adjs[i_eco]=diff_p
                
        # return adjustment factor
        return (eco_adjs)

def main():
    ################# FILE MANAGEMENT #######################
    
    start_loading = time.time()
    
    ####folder paths
    
    ## input folder (for now files)
    #input data annual
    dgvms_path = r"O:\H73700\3015_Furnaces-Couch\NPPpot\LPJ_5yrAvrg_negClip_eucAlloc_NN_30s\conly\Total.anpp.WaterStress.Conly_negFill_5yrAvrg_110.tif"
    dgvm_smooth_path =  r"O:\H73700\3015_Furnaces-Couch\NPPpot\LPJ_5yrAvrg_negClip_eucAlloc_Spline_30s\conly\Total.anpp.WaterStress.Conly_negFill_5yrAvrg_110.tif"
    
    # aux data annual
    nonprod_path = r"O:\H73700\3015_Furnaces-Couch\LUarea\2010NPS_km2.tif"
    land_path = r"O:\H73700\3015_Furnaces-Couch\LUarea\2010LAREA_km2.tif"
    
    ## input files static
    npp_proxy_path = r"C:\Users\floweidinger\Projekte\HANPP_Zeitreihe\DGVM_downscaling\aux_data\npp_proxy\npp_proxy.tif"
    ecoregions_path= r"C:\Users\floweidinger\Projekte\HANPP_Zeitreihe\DGVM_downscaling\aux_data\ecoregions\ecoregions.tif"
    
    #####open
    dgvm_ds = gdal.Open(dgvms_path)
    proxy_ds = gdal.Open(npp_proxy_path)
    ecoregions_ds = gdal.Open(ecoregions_path)
    nonprod_ds  = gdal.Open(nonprod_path)
    land_ds = gdal.Open(land_path)
    dgvm_smooth_ds = gdal.Open(dgvm_smooth_path)
    
    ######## Selecting subdata ########
    
    #### get x/y resolution in degrees
    #ulx, xres, xskew, uly, yskew, yres = americas_ref_ds.GetGeoTransform() # change back to america if used as bbox
    ulx, xres, xskew, uly, yskew, yres = proxy_ds.GetGeoTransform()
    
    #print("DEM res:",xres,  yres)
    #### bbox Dimensions in degrees
    #[latmin, lonmin, latmax, lonmax]
    #bbox = [-124, 48,-122,50] # Vancouver
    #bbox = [-126, 22,-71, 50] # USA 
    #bbox = [-180, -90,-34,90] # americas
    #bbox = [-115, 37,-108, 40] # Utah, US 
    bbox = [-76,-56,-63,-46] # Cap Hoorn
    #bbox = [ulx, uly-180, ulx+360, uly] # globe 
    
    #inv_geotransform = gdal.InvGeoTransform(americas_ref_ds.GetGeoTransform()) #to get x/y pixel dimensions from degrees # change back to america if used as bbox
    inv_geotransform = gdal.InvGeoTransform(proxy_ds.GetGeoTransform())
    #print("DEM ", inv_geotransform)
    
    #### convert lon lat degrees to x/y pixels for dataset
    _x0,_y0 = gdal.ApplyGeoTransform(inv_geotransform, bbox[0], bbox[1])
    _x1,_y1 = gdal.ApplyGeoTransform(inv_geotransform, bbox[2], bbox[3])
    x0 ,y0 = min(_x0, _x1), min(_y0, _y1)
    x1, y1 = max(_x0, _x1), max(_y0, _y1)
    
    #### Get Subset Data
    proxy = proxy_ds.ReadAsArray(int(x0), int(y0), int(x1-x0), int(y1-y0))
    dgvm = dgvm_ds.ReadAsArray(int(x0), int(y0), int(x1-x0), int(y1-y0))
    ecoregions = ecoregions_ds.ReadAsArray(int(x0), int(y0), int(x1-x0), int(y1-y0))
    ecoregions=ecoregions.astype(float)
    nonprod = nonprod_ds.ReadAsArray(int(x0), int(y0), int(x1-x0), int(y1-y0))
    
    end_loading = time.time()
    print("Loading time: ", end_loading-start_loading)
    
    start_clipping = time.time()
    ######## land Clip ########
    #load array
    land = land_ds.ReadAsArray(int(x0), int(y0), int(x1-x0), int(y1-y0))
    # set 0 to nan 
    land[land == 0] = np.nan 
    # create landmaskt with nan and 1 
    landmask_bin = land > 0 
    landmask = np.empty(land.shape)
    landmask[landmask_bin==False] = np.nan
    landmask[landmask_bin==True] = 1
    
    landmask_bin = None
    
    # clip other arrays with landmask
    proxy = proxy * landmask
    ecoregions = ecoregions * landmask
    dgvm = dgvm * landmask
    nonprod = nonprod * landmask
    
    landmask = None
    
    ###### pre calculation nonproductive area clip ########
    
    #every pixel that has 100% nonproductivity of underlying km2 (from landarea) is set to noData, otherwise it will be set to 1 
    #nonprod[nonprod != land] = 1 # every pixel with zero to some nonproductivity will be set to 1 to represent productive land 
    nonprod[nonprod == land] = np.nan # every pixel that has 100% nonprod to np.nan.
    nonprod[~np.isnan(nonprod)] = 1
    
    proxy = proxy * nonprod
    ecoregions = ecoregions * nonprod
    dgvm = dgvm * nonprod
    
    nonprod = None
    
    
    end_clipping = time.time()
    print("Clipping time: ", end_clipping-start_clipping)
    ################# CALCULATIONS #######################
    
    # proxy raster smoothing
    start_proxy = time.time()
    
    proxy_smooth = proxy_smoother(60,proxy,land,dgvm)
    
    end_proxy = time.time()
    print("Proxy map time: ", end_proxy-start_proxy)
    
    ###### dgvm Downscaling######
    # load smoothed dgvm
    dgvm_smooth = dgvm_smooth_ds.ReadAsArray(int(x0), int(y0), int(x1-x0), int(y1-y0))
    
    start_down = time.time()
    #Multiplication with smoothed DGVM NPP with generated factor raster (proxy_dev)
    dgvm_adj = dgvm_smooth*proxy_smooth
    
    proxy_smooth= None
    dgvm_smooth= None
    
    #del -inf values (check if thex still happen)
    dgvm_adj[np.where(dgvm_adj==float("-inf"))]=np.nan
    
    end_down = time.time()
    
    print("Downscaling time: ", end_down-start_down)
    ###### ecoregion total Sum readjustment ######
    
    start_readj_loop = time.time()
    # get unique ecoregion values 
    ecoids = np.unique(ecoregions)
    
    # create empty result array
    dgvm_readj = np.empty(dgvm_adj.shape)
    dgvm_readj.fill(np.nan)

    eco_adj = dict((el,0) for el in ecoids) # list with ecoregion adjustment percentage
    
    
    #### Multiproccesing ecoregion readjustment v2
    start_readj_mp_v2 = time.time()
    cores = mp.cpu_count()
    splitted_ecoids = np.array_split(ecoids, cores)

    g = partial(ecoreadj_loop, dgvm_eco=dgvm, dgvm_adj_eco=dgvm_adj, land_eco=land, ecoregions_eco= ecoregions)
    p = mp.Pool()
    
    eco_adj_mp = []
    eco_adj_mp_dict = {}
    
    eco_adj_mp = p.map(g,splitted_ecoids)
    
    
    for chunk in eco_adj_mp:
        if chunk is not None:
            # eco_adj_mp_dict = eco_adj_mp_dict | chunk # "|" operand apperently not working anymore in python 3.10 idk
            eco_adj_mp_dict = dict(list(eco_adj_mp_dict.items()) + list(chunk.items()))
            
            
    #  del nan values from dictionairy
    eco_adj_mp_dict = {k: eco_adj_mp_dict[k] for k in eco_adj_mp_dict if not isnan(k)}
    
    #print("MPv2, ecoadj: " ,eco_adj_mp)
    #print("MPv2, ecoadj Dictionairy : " ,eco_adj_mp_dict)
    
    # create new array for readjustd results
    dgvm_readj = np.empty(dgvm_adj.shape)
    dgvm_readj.fill(np.nan)
    
    #loop through ecoregion adjustment factors and multiply them to the dgvm 
    for eco_adj_f in eco_adj_mp_dict:
        dgvm_readj[np.where(ecoregions==eco_adj_f)] = dgvm_adj[np.where(ecoregions==eco_adj_f)]*eco_adj_mp_dict[eco_adj_f]
        
    end_readj_mp_v2 = time.time()
    print("Ecoregion readjust time multiproccesing: ", end_readj_mp_v2-start_readj_mp_v2)
    
    #dgvm = None
    #dgvm_adj = None
    #ecoregions = None
    #land= None
    
    ###### post readj nonproductive area clip ########
    #nonprod = nonprod_ds.ReadAsArray(int(x0), int(y0), int(x1-x0), int(y1-y0))
    #dgvm_readj = dgvm_readj * nonprod

    ########## OutPuts ###################
    
    plt.imshow(dgvm_readj)
    print("max value:" , np.nanmax(dgvm_readj))

    outpath = r"C:\Users\floweidinger\Projekte\HANPP_Zeitreihe\DGVM_downscaling\output_data//"
    '''
    ##### Ecoregion Readjust Factor OutPut ######
    import csv
    eco_adj_path = outpath + "test.csv"
    
    with open(eco_adj_path, 'w') as csvfile:
        for key in eco_adj.keys():
            csvfile.write("%s,%s\n"%(key,eco_adj[key]))
        
    ##### GeoTIFF OutPut ######
    tiffpath= outpath + "test.tif"
    
    ## only outputs the bbox
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(tiffpath,dgvm_readj.shape[1],dgvm_readj.shape[0],1,gdal.GDT_Float32)
    geotransform=(bbox[0],xres,0,bbox[3],0,yres) ##lat lon res from bbox, skew set 0
    ds.SetGeoTransform(geotransform)
    ds.GetRasterBand(1).WriteArray(dgvm_readj)
    ds = None
    
    '''
    #### close datasets
    #dgvm_readj = None

if __name__ == "__main__":
  main()


