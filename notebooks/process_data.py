import rasterio
from rasterio.warp import reproject, Resampling
import os
import glob
import shutil
import numpy as np



def run_resampling(data_dir='DATA'):
    '''Resamples a set of raster to ahve the same bounds'''
    amp_files = glob.glob(data_dir + os.sep + '*' + os.sep + '*amp.tif')
    unw_files = glob.glob(data_dir + os.sep + '*' + os.sep + '*unw_phase.tif')
    ph_files = glob.glob(data_dir + os.sep + '*' + os.sep + '*wrapped_phase.tif')
    cor_files = glob.glob(data_dir + os.sep + '*' + os.sep + '*corr.tif')
    dem_files = glob.glob(data_dir + os.sep + '*' + os.sep + '*dem.tif')
    lv_theta_files = glob.glob(data_dir + os.sep + '*' + os.sep + '*lv_theta.tif')
    lv_phi_files = glob.glob(data_dir + os.sep + '*' + os.sep + '*lv_phi.tif')
    inc_files = glob.glob(data_dir + os.sep + '*' + os.sep + '*inc_map.tif')
    inc_ell_files = glob.glob(data_dir + os.sep + '*' + os.sep + '*inc_map_ell.tif')
    mask_files = glob.glob(data_dir + os.sep + '*' + os.sep + '*water_mask.tif')
    vert_disp_files = glob.glob(data_dir + os.sep + '*' + os.sep + '*vert_disp.tif')
    los_disp_files = glob.glob(data_dir + os.sep + '*' + os.sep + '*los_disp.tif')

    if len(unw_files)==0:
        raise RuntimeError('I did not find any unwrapped files in {}'.format(data_dir))

    ref_file = 0
    with rasterio.open(unw_files[ref_file], 'r') as f:
        ref_crs = f.crs
        ref_bounds = f.bounds
        ref_width = f.width
        ref_height = f.height
        ref_ndv = f.nodata

    kwupdate = {
        'crs': ref_crs,
        'bounds': ref_bounds,
        'width': ref_width,
        'height': ref_height,
        'nodata': ref_ndv
        }

    for k, f in enumerate(unw_files):
        af = find_matching_file(amp_files, f)
        ph = find_matching_file(ph_files, f)
        cf = find_matching_file(cor_files, f)
        df = find_matching_file(dem_files, f)
        lvf = find_matching_file(lv_theta_files, f)
        lpf= find_matching_file(lv_phi_files, f)
        incf = find_matching_file(inc_files, f)
        ief = find_matching_file(inc_ell_files, f)
        mf = find_matching_file(mask_files, f)
        vf = find_matching_file(vert_disp_files, f)
        lf = find_matching_file(los_disp_files, f)

        update_file(af, kwupdate)
        update_file(f, kwupdate)
        update_file(ph, kwupdate)
        update_file(cf, kwupdate)
        update_file(df, kwupdate)
        update_file(lvf, kwupdate)
        update_file(lpf, kwupdate)
        update_file(incf, kwupdate)
        update_file(ief, kwupdate)
        update_file(mf, kwupdate)
        update_file(vf, kwupdate)
        update_file(lf, kwupdate)



def update_file(orig_file, kwupdate):
    # See here: https://hatarilabs.com/ih-en/how-to-reproject-single-and-multiple-rasters-with-python-and-rasterio-tutorial
    if orig_file is None:
        return

    # first open the original file
    srcRst = rasterio.open(orig_file)
    options = srcRst.meta.copy()
    options.update(kwupdate)

    # Now create a temporary new file
    tmp_file = 'tmp.tif'
    dstRst = rasterio.open(tmp_file, 'w', **options)

    # write each band in the old file to the new file
    for i in range(1, srcRst.count + 1):
        reproject(
            source=rasterio.band(srcRst, i),
            destination=rasterio.band(dstRst, i),
            src_crs=srcRst.crs,
            dst_crs=srcRst.crs,
            resampling=Resampling.nearest,
        )

    # close both files
    dstRst.close()
    srcRst.close()

    # Copy the new file to the old file location
    os.replace('tmp.tif', orig_file)


def find_matching_file(flist, f):
    # format: S1AA_20200920T001203_20201002T001203_***
    parts1 = os.path.basename(f).split('_')
    for f2 in flist:
        parts = os.path.basename(f2).split('_')
        if (parts1[1] == parts[1]) & (parts1[2] == parts[2]):
            return f2
    return None


def plot_extents(data_dir):
    unw_files = glob.glob(data_dir + os.sep + '*' + os.sep + '*unw_phase.tif')
    NS_bounds = []
    for f in unw_files:
        with rasterio.open(f) as F:
            NS_bounds.append(F.bounds[:2])

    de = np.array(NS_bounds).mean(axis=0)

    for k, pair in enumerate(NS_bounds):
        plt.plot([k,k], [pair[0]-de[0], pair[1]-de[1]], '-k')
    plt.ylim([de[0] - 100, de[1] + 100])

    plt.savefig('Network_extents.png')
    plt.close('all')


if __name__=='__main__':
    run_resampling(data_dir='.')

