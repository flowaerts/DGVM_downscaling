# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 15:22:07 2022

@author: floweidinger
"""
import time
import numpy as np

def proxy_smoother(matrix30s_size, proxy, land, dgvm):

    ###### NPP proxy map smoothing ###### 
    #matrix30s_size = 60 #in 30min are 60x30sec Pixels
    
    #for offset in pbar(range(0,matrix30s_size)): #for loop with progress bar
    for offset in range(0,matrix30s_size):
        
        proxy_dev = np.empty(proxy.shape)
        proxy_dev.fill(0)
        
        for y in range(offset,proxy.shape[0], matrix30s_size): #iterates with 30min steps in the 30sec raster. 
            for x in range(offset, proxy.shape[1], matrix30s_size):
                
                #calc of 30min value (area weighted)
                sum_prod = np.nansum(
                    np.multiply(
                        proxy[y:y+matrix30s_size,x:x+matrix30s_size],
                        land[y:y+matrix30s_size,x:x+matrix30s_size]
                        ))
                
                sqkm_sum = np.nansum(land[y:y+matrix30s_size,x:x+matrix30s_size][~np.isnan(proxy[y:y+matrix30s_size,x:x+matrix30s_size])])
                proxy_30min =sum_prod/sqkm_sum
                
                #deviations of 30sec proxy values from pointer 30min value
                proxy_dev[y:y+matrix30s_size,x:x+matrix30s_size] = proxy[y:y+matrix30s_size,x:x+matrix30s_size]/proxy_30min
                
        if offset == 0:
            proxy_offset_collector = proxy_dev
        else:
            proxy_offset_collector = proxy_offset_collector + proxy_dev
    
    proxy_dev = None
    proxy = None
    
    proxy_smooth = np.empty(dgvm.shape)
    proxy_smooth.fill(0)
    
    #edge boundaries
    for i in range(0,matrix30s_size):
        proxy_smooth[i,matrix30s_size:] = proxy_offset_collector[i,matrix30s_size:]/(i+1)
        proxy_smooth[matrix30s_size:,i] = proxy_offset_collector[matrix30s_size:,i]/(i+1)
    
    mean_index = np.indices((matrix30s_size,matrix30s_size))
    mean_factor = np.empty(mean_index.shape)
    mean_factor = np.minimum(mean_index[0], mean_index[1])+1
    
    proxy_smooth[:matrix30s_size,:matrix30s_size] = proxy_offset_collector[:matrix30s_size,:matrix30s_size]/mean_factor 
    proxy_smooth[matrix30s_size:,matrix30s_size:] = proxy_offset_collector[matrix30s_size:,matrix30s_size:]/matrix30s_size
    
    proxy_offset_collector = None
    
    return proxy_smooth
