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


def symmetric_cf_levels(da, nlevels):
    '''
    Creates an array of contour levels symmetric about 0.
    The min/max limit is given by the median+3*stddev

    Parameters:
    -----------
    da : xr.DataArray
        Data array

    Returns:
    --------
    levels : np.ndarray
        Array of contour levels

    Author: Ben Buchovecky
    '''
    flattened = da.values.flatten()
    lim = np.nanmedian(abs(flattened))+3*np.nanstd(flattened)
    levels = np.linspace(-lim, lim, nlevels)
    return levels


def weighted_average(da, weights, lat_bnds=[-90,90]):
    '''
    Calculates a weighted average along the latitude and longitude
    dimensions within a given latitude range from an array of 
    gridcell weights.
    
    Parameters:
    -----------
    da : xr.DataArray
        Data array
    weights : xr.DataArray or np.array
        Array of gridcell weights
    lat_bnds : list
        Minimum and maximum latitude bounds. The default bounds are
        the entire globe [-90,90]N

    Returns:
    --------
    weighted : xr.DataArray
        Weighted spatial average.

    Author: Ben Buchovecky
    '''
    if lat_bnds != [-90,90]:
        weights = weights.where((weights.lat>=lat_bnds[0]) & (weights.lat<lat_bnds[1]))
        weights = weights / weights.mean(dim=['lon','lat'])

    weighted = (da * weights).where(weights != 0)
    weighted = weighted.where((da.lat>=lat_bnds[0]) & (da.lat<lat_bnds[1]))
    weighted = weighted.mean(dim=['lat','lon'])
    return weighted
