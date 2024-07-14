import datetime
import glob
import h5py
import os
import rasterio
import re

import numpy as np
from osgeo import gdal


def main(ifgList,refCenter=None,refSize=None):
    '''
    Read in a list of interferograms and create a time-series and velocity map. 
    '''
    datePairs, dates = getDates(ifgList)
    fracDates = np.array([dt2fracYear(d) for d in dates])
    G = makeG(dates, datePairs)
    data = getData(ifgList,1)
    data = dereference(data, taxis=0,refCenter=refCenter,refSize=refSize)
    tsArray = makeTS(G, data, fracDates)
    vel = findMeanVel(tsArray, fracDates, 0)
    vel = convertRad2meters(vel)
    writeTS2HDF5(tsArray, fracDates,vel)


def getDates(ifgList):
    datePairs = []
    unique_dates = []

    for ifg in ifgList:
        ds1, ds2 = re.search("\d{8}T*\d*_\d{8}T*\d*", ifg).group().split('_') # grab the date(time) strings
        d1, d2 = [datetime.datetime.strptime(re.sub('T\d{6}', '', d), '%Y%m%d') for d in (ds1, ds2)] # remove any time signature
        if d1 not in unique_dates:
            unique_dates.append(d1)
        if d2 not in unique_dates:
            unique_dates.append(d2)
        datePairs.append((d1,d2))

    unique_dates.sort()

    return datePairs, np.array(unique_dates)


def dt2fracYear(date):
    import datetime as dt
    import time

    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    # check that the object is a datetime
    year = date.year

    startOfThisYear = dt.datetime(year=year, month=1, day=1)
    startOfNextYear = dt.datetime(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration
    date_frac = date.year + fraction

    return date_frac


def makeG(dates, pairs):
    '''
    Create a time-series G-matrix. "dates" should be sorted ascending.
    '''
    G = np.zeros((len(pairs), len(dates)))
    for k, (d1, d2) in enumerate(pairs):
        index1= dates == d1
        index2 = dates == d2
        G[k,index1] = -1
        G[k,index2] = 1
    return G


def readRaster(filename, band_num = None):
    '''
    Read a GDAL VRT file and return its attributes
    '''
    try:
        ds = gdal.Open(filename, gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError('readRaster: cannot find file {}'.format(filename))
    except Exception as e:
        print(e)
        raise RuntimeError('readRaster: cannot find file {}'.format(filename))

    xSize = ds.RasterXSize
    ySize = ds.RasterYSize
    geoProj = ds.GetProjection()
    trans = ds.GetGeoTransform()

    if xSize==0:
        raise RuntimeError('readRaster: xSize is zero, cannot continue')
    if ySize==0:
        raise RuntimeError('readRaster: ySize is zero, cannot continue')

    Nbands = ds.RasterCount
    if band_num is None:
        band_num = 1
        print('Using band one for dataType')

    dType = ds.GetRasterBand(band_num).DataType
    noDataVal = ds.GetRasterBand(band_num).GetNoDataValue()
    ds = None

    return xSize, ySize, dType, geoProj, trans, noDataVal, Nbands


def getData(ifgList, band_num):
    pix=[]
    for ifg in ifgList:
        pix.append(readIFG(ifg,band_num=band_num))
    pix = np.array(pix)
    return pix 


def dereference(array, taxis=0,refCenter = None, refSize = None):
    #taxis must be 0
    if taxis!=0:
        raise RuntimeError('taxis must be zero')
    
    if refCenter is None:
        nshape = array.shape[1:]
        refCenter = [d//2 for d in nshape]
        print('Reference region is centered on {}/{}'.format(refCenter[0],refCenter[1]))
    if refSize is None:
        refSize = 10
        print('Reference region is {} square pixels'.format(refSize**2))

    row1,row2,col1,col2=refCenter[0]-refSize//2,refCenter[0]+refSize//2,refCenter[1]-refSize//2,refCenter[1]+refSize//2
    for k in range(array.shape[taxis]):
        array[k,...] - np.nanmean(array[k,list(range(row1,row2)),list(range(col1,col2))])
    return array


def makeTS(G, array, fracDates, taxis = 0):
    in_shape = array.shape
    Nt = G.shape[-1]
    nshape = tuple(s for i,s in enumerate(in_shape) if i!=taxis)
    narray = np.swapaxes(array,0,taxis)
    flat_array = array.reshape((in_shape[taxis],)+(np.prod(nshape),))
    that,res,rank,s = np.linalg.lstsq(G,flat_array, rcond=None) 
    out_array = that.reshape((Nt,)+nshape)
    return out_array
    

def findMeanVel(array, t, taxis=0):
    in_shape = array.shape
    Nt = in_shape[taxis]
    nshape = tuple(s for i,s in enumerate(in_shape) if i!=taxis)
    narray = np.swapaxes(array,0,taxis)
    flat_array = array.reshape((Nt,)+(np.prod(nshape),))

    G = np.ones((in_shape[taxis],)+(2,))
    G[:,1] = t - t[0]

    that,res,rank,s = np.linalg.lstsq(G,flat_array, rcond=None) 
    out_vel = that.reshape((2,)+nshape)
    return out_vel[1,...]
    

def convertRad2meters(vel, lam=0.056):
    '''
    Convert radians to mm
    '''
    return vel/(4*np.pi/lam)


def writeTS2HDF5(array, dates, vel,filename='ts.h5'):
    with h5py.File(filename,'w') as f:
        f['ts'] = array
        f['dates'] = dates
        f['vel']=vel
    print('Finished writing {} to disk'.format(filename))


def readIFG(ifg, band_num=1, xstart=0, ystart=0, xStep=None,yStep=None):
    ds = gdal.Open(ifg)
    if xStep is None:
        data = ds.GetRasterBand(band_num).ReadAsArray()
    else:
        data = ds.GetRasterBand(band_num).ReadAsArray(xstart, ystart, xStep,yStep)
    del ds
    return data


def gdal_open(fname, returnProj=False, userNDV=None, band=None):
    '''
    Reads a rasterio-compatible raster file and returns the data and profile
    '''
    if rasterio is None:
        raise ImportError('RAiDER.utilFcns: rio_open - rasterio is not installed')

    if os.path.exists(fname + '.vrt'):
        fname = fname + '.vrt'

    with rasterio.open(fname) as src:
        profile = src.profile

        # For all bands
        nodata = src.nodatavals

        # If user requests a band
        if band is not None:
            ndv = nodata[band - 1]
            data = src.read(band).squeeze()
            nodataToNan(data, [userNDV, nodata[band - 1]])

        else:
            data = src.read().squeeze()
            if data.ndim > 2:
                for bnd in range(data.shape[0]):
                    val = data[bnd, ...]
                    nodataToNan(val, [userNDV, nodata[bnd]])
            else:
                nodataToNan(data, list(nodata) + [userNDV])


        if data.ndim > 2:
            dlist = []
            for k in range(data.shape[0]):
                dlist.append(data[k,...].copy())
            data = dlist

    if not returnProj:
        return data

    else:
        return data, profile
    

def nodataToNan(inarr, listofvals):
    """
    Setting values to nan as needed
    """
    inarr = inarr.astype(float) # nans cannot be integers (i.e. in DEM)
    for val in listofvals:
        if val is not None:
            inarr[inarr == val] = np.nan
        

def getTSfromIFGs(i,j,ifg, band_num=1):
    ds = gdal.Open(ifg)
    value = ds.GetRasterBand(band_num).ReadAsArray(i, j, 1, 1)
    del ds
    return value[0][0]


if __name__=='__main__':
    import glob
    ifgList = glob.glob('*_unw.vrt')
    main(ifgList)
