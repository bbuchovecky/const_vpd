import os
import numpy as np
import xarray as xr
import scipy.signal as sps
import pickle as pkl
from warnings import catch_warnings,simplefilter

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.path as mpath

import cartopy
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point


def add_cartopy_gridlines(ax,
                          mapproj,
                          lat=[-60, -30, 0, 30, 60],
                          lon=[-120, -60, 0, 60, 120, 180],
                          gridstyle={'linewidth':0.5,
                                     'linestyle':':',
                                     'color':'k'}):
    '''
    Adds gridlines to a Cartopy map.
    
    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        Cartopy geoaxis object.
    mapproj : cartopy.crs.PROJECTION
        Cartopy map projection class.
    lat : list
        List of latitude grid coordinates.
    lon : list
        List of longitude grid coordinates.
    gridstyle : dict
        Dictionary of line properties for gridlines.
        
    Returns:
    --------
    n/a
    
    Author: Ben Buchovecky
    '''
    
    gl = ax.gridlines(linewidth=gridstyle['linewidth'],
                      linestyle=gridstyle['linestyle'],
                      color=gridstyle['color'])
    gl.ylocator = mticker.FixedLocator(lat)
    gl.xlocator = mticker.FixedLocator(lon)

    _, y_btm = mapproj.transform_point(0, -90, ccrs.Geodetic())
    _, y_top = mapproj.transform_point(0, 90, ccrs.Geodetic())
    ax.set_ylim(y_btm, y_top)
    

def cyclic_contourf(ax,
                    da,
                    **kwargs):
    '''
    Adds a cyclic point and creates a filled contour plot.
    
    Parameters:
    -----------
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        Cartopy geoaxis object.
    da : xarray.DataArray
        DataArray with lat and lon coordinates.
    
    Returns:
    --------
    cf : cartopy.mpl.contour.GeoContourSet
        Contourf plot with an added cyclic point.
    
    Author: Ben Buchovecky
    '''
    
    data,lon = add_cyclic_point(da, coord=da.lon)
    cf = ax.contourf(lon, da.lat, data, **kwargs)
    return cf


def symmetric_y_axis(ax):
    '''
    Sets the limits on the y-axis such that the min and max are
    symmetric about zero. This is useful for plots of differences
    between two datasets.
    
    Parameters:
    -----------
    ax : matplotlib.axes
        Matplotlib axis object
        
    Returns:
    --------
    n/a
    
    Author: Ben Buchovecky
    '''
    
    ax.set_ylim(-max(np.abs(ax.get_ylim())), max(np.abs(ax.get_ylim())))
    

def coslat_area_avg(da):
    '''
    Computes the cosine(latitude) weighted average.

    Parameters:
    -----------
    da : xr.DataArray
        Lat/lon gridded data array
        
    Returns:
    --------
    avg : xr.DataArray
        Cosine(latitude) weighted average data array
        
    Author: Ben Buchovecky
    '''
    
    lat = da.lat
    return (da.mean(dim='lon')*np.cos(np.deg2rad(lat))).mean(dim='lat')


def coslat_weight(da):
    '''
    Weights a lat/lon gridded array with cosine(latitude)

    Parameters:
    -----------
    da : xr.DataArray
        Lat/lon gridded data array
        
    Returns:
    --------
    avg : xr.DataArray
        Cosine(latitude) weighted data array
        
    Author: Ben Buchovecky
    '''
    
    lat = da.lat
    return da*np.cos(np.deg2rad(lat))