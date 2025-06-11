import xarray as xr
import uxarray as ux
import numpy as np
from tqdm import tqdm
import huracanpy
import pandas as pd

from tcpyPI import pi

import intake
from easygems import healpix as egh

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # don't warn us about future package conflicts

import healpy as hp

from dask_mpi import initialize
initialize()

from dask.distributed import Client
client = Client()

#WARNING! some models have 3D only when called with different keys!
#sim_name = "um_glm_n2560_RAL3p3"
sim_name = "ifs_tco3999-ng5_deepoff"
zoom_level = 7
time_res=6
out_dir = "/work/bb1153/b383007/hk25-hamburg/out_data/"
complete_fields = False
margin_deg = 8
tid_start = 20

if complete_fields:
    out_dir = out_dir + "complete_fields/"
else:
    out_dir = out_dir + "only_tracks/"

current_location = "online"

def get_pi(ts, psl, p, t, q):
    #result = pi(ts, psl, p, t, q, 
    #            kwargs=dict(CKCD=CKCD, ascent_flag=0, 
    #                        diss_flag=1, ptop=50, miss_handle=1))
    result = xr.apply_ufunc(
        pi,
        ts, psl, p, t, q,
        kwargs=dict(CKCD=CKCD, ascent_flag=0, diss_flag=1, ptop=50, 
                    miss_handle=1),
        input_core_dims=[
            [], [], [dim, ], [dim, ], [dim, ],
        ],
        output_core_dims=[
            [], [], [], [], []
        ],
        vectorize=True,
        output_dtypes=[ts.dtype, ts.dtype, ts.dtype, ts.dtype, ts.dtype],
    )
    vmax, pmin, ifl, t0, otl = result
    return vmax, pmin, t0, otl, ts

#allows to save uxgrid
def sanitize_attrs(ds):
    """Convert boolean attributes to int or str for NetCDF compatibility."""
    new_attrs = {}
    for k, v in ds.attrs.items():
        if isinstance(v, (bool, np.bool_)):
            new_attrs[k] = int(v)  # or str(v)
        else:
            new_attrs[k] = v
    ds.attrs = new_attrs
    return ds


#def main():
print(f"Simulation name: {sim_name}\n Zoom level: {zoom_level}\n Time resolution: {time_res}H")

#cluster = LocalCluster(n_workers=n_workers, 
#                       threads_per_worker=n_cpu // n_workers,
#                       memory_limit=memory_limit,
#                       processes=processes,
#                       dashboard_address=46861)

cat = intake.open_catalog("https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml")[current_location]

if "um_" in sim_name:
    ds2d = cat[sim_name](zoom=zoom_level).to_dask()

    vnames = {"sst":"ts", "psl":"psl", "t":"ta", "q":"hus", "cell":"cell"}

    ds3d = cat[sim_name](zoom=zoom_level, time="PT3H").to_dask()
    
    dim="pressure"
    
if "icon_" in sim_name:
    ds2d = cat[sim_name](zoom=zoom_level, time='PT6H', time_method='inst').to_dask()
    
    vnames = {"sst":"ts", "psl":"psl", "t":"ta", "q":"hus", "cell":"cell"}

    ds3d = ds2d.copy()

    dim="pressure"
    ds3d = ds3d.assign_coords(pressure=ds3d[dim]/100)
if "ifs_" in sim_name:
    ds2d = cat[sim_name](zoom=zoom_level, dim="2D").to_dask()
    
    vnames = {"sst":"sst", "psl":"msl", "t":"t", "q":"q", "cell":"value"}

    ds3d = cat[sim_name](zoom=zoom_level, dim="3D").to_dask()

    dim="level"

ds3d = ds3d.sortby(dim, ascending=False)
time= ds2d.time.where(ds2d.time.dt.hour % time_res == 0, drop=True)

ds2d = ds2d.rename({vnames["cell"]: "cell"})
ds3d = ds3d.rename({vnames["cell"]: "cell"})

#rechunk only if necessary
#ds2d = ds2d.chunk(chunks={"time":100})#, "cell":2048})
#ds3d = ds3d.chunk(chunks={"time":100})#, "cell":2048})

ds2d = ux.UxDataset.from_healpix(ds2d, pixels_only=False)
ds3d = ux.UxDataset.from_healpix(ds3d, pixels_only=False)

ts = ds2d[vnames["sst"]]
psl = ds2d[vnames["psl"]]
t = ds3d[vnames["t"]]
q = ds3d[vnames["q"]]

p=ds3d[dim]
CKCD=0.9
    
if not "icon_" in sim_name:
    ftrack = f"../TC_tracks/{sim_name}.csv"
else:
    ftrack = f"../TC_tracks/2D/{sim_name}.csv"
tracks = huracanpy.load(ftrack)

if not complete_fields:
    unique_tracks = np.unique(tracks.track_id)
    #for tid in unique_tracks:
    for tid in tqdm(unique_tracks, desc="Tracks", position=0):
        if tid<tid_start:
            continue
        track = tracks.where(tracks.track_id==tid, drop=True)

        track_time = track['time']
        track_time_res = pd.to_timedelta(track_time[1].values - track_time[0].values)

        #loop over distinct points on the track
        for i in tqdm(range(len(track['time'])), 
                      desc="Position on track", position=1, leave=False):
            cur_time  = track['time'][i]
            start = pd.Timestamp(cur_time.values) - pd.Timedelta(days=3)
            end   = pd.Timestamp(cur_time.values) + pd.Timedelta(days=3)
            track_time = pd.date_range(start, end, freq=track_time_res)
            track_time = np.clip(track_time, ds2d.time.values[0], ds2d.time.values[-1])
            
            track_lat = track['lat'][i]
            track_lon = track['lon'][i]
            
            min_lat = float(track_lat) - margin_deg
            max_lat = float(track_lat) + margin_deg
            min_lon = float(track_lon) - margin_deg
            max_lon = float(track_lon) + margin_deg

            if max_lon > 180:
                max_lon = max_lon - 360
            elif min_lon <-180:
                min_lon = 360 - abs(min_lon)
            #else:
            #    continue #avoid redoing track steps where this condition did not occur

            if max_lat > 90:
                max_lax = 90
            elif min_lat <-90:
                min_lat = -90
        
            lat_slice  = (min_lat, max_lat)
            lon_slice  = (min_lon, max_lon)

            grid = ts.subset.bounding_box(lon_slice, lat_slice).uxgrid 
            path_grid = out_dir+f'{sim_name}_pi_zoom_{zoom_level:02d}_track_{tid:04d}_pos_{i:04d}_grid'+'.nc'
            grid = sanitize_attrs(grid.to_xarray())
            grid.to_netcdf(path_grid)
            
            track_time_unique = np.unique(track_time)
            
            ts_tid  = ts.sel(time=track_time_unique).subset.bounding_box(lon_slice, lat_slice).load()
            psl_tid = psl.sel(time=track_time_unique).subset.bounding_box(lon_slice, lat_slice).load()
            t_tid = t.sel(time=track_time_unique).subset.bounding_box(lon_slice, lat_slice).load()
            q_tid = q.sel(time=track_time_unique).subset.bounding_box(lon_slice, lat_slice).load()   
            
            ts_max = ts_tid.max("time")

            #vmax, pmin, t0, otl, sst = get_pi(ts_tid -273.15, 
            #                                  psl_tid /100, p, 
            #                                  t_tid - 273.15, 
            #                                  q_tid * 1000)
            
            futures, futures2 = [], []
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'Sending large graph of size', UserWarning)
                for ti in track_time:
                    #vmax, pmin, t0, otl = get_pi(ts_tid[0], psl_tid[0], p, t_tid[0], q_tid[0])

                    #ts_tid = ts.sel(time=ti, n_face=faces).load()#.subset.bounding_box(lon_slice, lat_slice).load()
                    #psl_tid = psl.sel(time=ti, n_face=faces).load()#.subset.bounding_box(lon_slice, lat_slice).load()
                    #t_tid = t.sel(time=ti, n_face=faces).sortby(dim, ascending=False).load()#.subset.bounding_box(lon_slice, lat_slice).load()
                    #q_tid = q.sel(time=ti, n_face=faces).sortby(dim, ascending=False).load()#.subset.bounding_box(lon_slice, lat_slice).load()

                    ts_tid_ti = ts_tid.sel(time=ti)
                    ps_tid_ti = psl_tid.sel(time=ti)
                    t_tid_ti = t_tid.sel(time=ti)
                    q_tid_ti = q_tid.sel(time=ti)
                    
                    futures.append(client.submit(get_pi, 
                                                 ts_tid_ti -273.15, 
                                                 ps_tid_ti/100, p, 
                                                 t_tid_ti - 273.15, 
                                                 q_tid_ti * 1000))
                    futures2.append(client.submit(get_pi, 
                                                 ts_max -273.15, 
                                                 ps_tid_ti/100, p, 
                                                 t_tid_ti- 273.15, 
                                                 q_tid_ti * 1000))
            results = client.gather(futures)
            results2 = client.gather(futures2)
    
            # out_ds["vmax"][i] = ux.UxDataArray(vmax, uxgrid=grid)
            # out_ds["pmin"][i] = ux.UxDataArray(pmin, uxgrid=grid)
            # out_ds["t0"][i] = ux.UxDataArray(t0, uxgrid=grid)
            # out_ds["otl"][i] = ux.UxDataArray(otl, uxgrid=grid)
            # out_ds["sst"][i] = ux.UxDataArray(ts_tid[i], uxgrid=grid)
            
            out_ds = xr.Dataset()
            out_ds["vmax"] = xr.concat([item[0] for item in results], dim="time")
            out_ds["pmin"] = xr.concat([item[1] for item in results], dim="time")
            out_ds["t0"]   = xr.concat([item[2] for item in results], dim="time")
            out_ds["otl"]  = xr.concat([item[3] for item in results], dim="time")
            out_ds["sst"]  = xr.concat([item[4] for item in results], dim="time")

            out_ds["vmax_fixts"] = xr.concat([item[0] for item in results2], dim="time")
            out_ds["pmin_fixts"] = xr.concat([item[1] for item in results2], dim="time")
            out_ds["t0_fixts"]   = xr.concat([item[2] for item in results2], dim="time")
            out_ds["otl_fixts"]  = xr.concat([item[3] for item in results2], dim="time")
            #out_ds["vmax"] = vmax
            #out_ds["pmin"] = pmin
            #out_ds["t0"]   = t0
            #out_ds["otl"]  = otl
            #out_ds["sst"]  = sst

            # add names and units to the structure
            out_ds.vmax.attrs['standard_name'],out_ds.vmax.attrs['units']='Maximum Potential Intensity','m/s'
            out_ds.pmin.attrs['standard_name'],out_ds.pmin.attrs['units']='Minimum Central Pressure','hPa'
            #out_ds.ifl.attrs['standard_name']='pyPI Flag'
            out_ds.t0.attrs['standard_name'],out_ds.t0.attrs['units']='Outflow Temperature','K'
            out_ds.otl.attrs['standard_name'],out_ds.otl.attrs['units']='Outflow Temperature Level','hPa'
            out_ds.sst.attrs['standard_name'],out_ds.sst.attrs['units']='Sea surface temperature','K'

            out_ds.vmax_fixts.attrs['standard_name'],out_ds.vmax_fixts.attrs['units']='Maximum Potential Intensity, max SST on track','m/s'
            out_ds.pmin_fixts.attrs['standard_name'],out_ds.pmin_fixts.attrs['units']='Minimum Central Pressure, max SST on track','hPa'
            #out_ds.ifl.attrs['standard_name']='pyPI Flag'
            out_ds.t0_fixts.attrs['standard_name'],out_ds.t0_fixts.attrs['units']='Outflow Temperature, max SST on track','K'
            out_ds.otl_fixts.attrs['standard_name'],out_ds.otl_fixts.attrs['units']='Outflow Temperature Level, max SST on track','hPa'

            out_ds["track_id"] = tid

            path = out_dir+f'{sim_name}_pi_zoom_{zoom_level:02d}_track_{tid:04d}_pos_{i:04d}'+'.nc'
            out_ds.to_netcdf(path)
            del out_ds, futures
else:
    def process_chunk(ts, psl, p, t, q):
        return get_pi(ts, psl, p, t, q)

    if "ifs_" in sim_name:
        ts = ts.drop_vars(["lat", "lon"])
        psl = psl.drop_vars(["lat", "lon"])
        t = t.drop_vars(["lat", "lon"])
        q = q.drop_vars(["lat", "lon"])
    
    chunk_size = 25  # Adjust based on available memory or number of workers
    lat_chunks = [slice(i, i + chunk_size) for i in range(0, len(lat), chunk_size)]
    for ti in tqdm(range(len(time))):
        ts_ti = ts.sel(time=time[ti]).load()[pix] - 273.15
        psl_ti = psl.sel(time=time[ti]).load()[pix] / 100
        t_ti = t.sel(time=time[ti]).load()[:, pix] - 273.15
        q_ti = q.sel(time=time[ti]).load()[:, pix] * 1000
    #    if "icon_" in sim_name:
    #        psl_i = psl_i/100
        
        futures = []
        for chunk in lat_chunks:
            ts_cti = ts_ti.isel(lat=chunk)
            psl_cti = psl_ti.isel(lat=chunk)
            t_cti = t_ti.isel(lat=chunk)
            q_cti = q_ti.isel(lat=chunk)
    
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'Sending large graph of size', UserWarning)
                futures.append(client.submit(process_chunk, ts_cti, psl_cti, p, t_cti, q_cti))
            
        results = np.array(client.gather(futures), dtype=object)
    
        # Initialize with empty (NaN) arrays for each variable
        shape = (1, len(lat), len(lon))
        empty_arr = np.full(shape, np.nan, dtype=np.float32)
        dims = ["time", "lat", "lon"]
        
        out_ds = xr.Dataset({
                'vmax': (dims, empty_arr.copy()), 
                'pmin': (dims, empty_arr.copy()),
           #     'ifl': (dims, empty_arr),
                't0': (dims, empty_arr.copy()),
                'otl': (dims, empty_arr.copy()),
           #     'sst': (dims, empty_arr),
           #     't': (dims, empty_arr),
           #     'q': (dims, empty_arr),
           #     'msl': (dims, empty_arr),
                },
                coords={
                "time": [time[ti].values],
                "lat": lat,
                "lon": lon}
        )
        
        # add names and units to the structure
        out_ds.vmax.attrs['standard_name'],out_ds.vmax.attrs['units']='Maximum Potential Intensity','m/s'
        out_ds.pmin.attrs['standard_name'],out_ds.pmin.attrs['units']='Minimum Central Pressure','hPa'
        #out_ds.ifl.attrs['standard_name']='pyPI Flag'
        out_ds.t0.attrs['standard_name'],out_ds.t0.attrs['units']='Outflow Temperature','K'
        out_ds.otl.attrs['standard_name'],out_ds.otl.attrs['units']='Outflow Temperature Level','hPa'
    
        out_ds["vmax"].loc[dict(time=time[ti])] = xr.concat(results[:,0], dim="lat").values
        out_ds["pmin"][0, :, :] = xr.concat(results[:,1], dim="lat").values
        out_ds["t0"]  [0, :, :] = xr.concat(results[:,2], dim="lat").values
        out_ds["otl"] [0, :, :] = xr.concat(results[:,3], dim="lat").values
        #results = [xr.concat(results[i], dim="lat") for i in range(len(out_ds))]
        
        out_ds.to_netcdf(out_dir+f'{sim_name}_pi_{time[ti].values}'+'.nc')
client.close()



    