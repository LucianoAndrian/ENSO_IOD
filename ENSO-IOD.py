"""
ENSO vs IOD Composiciones temporal
"""
import xarray as xr
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import os
from matplotlib import colors
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from ENSO_IOD_Funciones import Nino34CPC
from ENSO_IOD_Funciones import DMI
from ENSO_IOD_Funciones import MultipleComposite
from ENSO_IOD_Funciones import Plots

def OpenDatasets(name, interp=False):
    pwd_datos = '/datos/luciano.andrian/ncfiles/'
    def ChangeLons(data, lon_name='lon'):
        data['_longitude_adjusted'] = xr.where(
            data[lon_name] < 0,
            data[lon_name] + 360,
            data[lon_name])

        data = (
            data
                .swap_dims({lon_name: '_longitude_adjusted'})
                .sel(**{'_longitude_adjusted': sorted(data._longitude_adjusted)})
                .drop(lon_name))

        data = data.rename({'_longitude_adjusted': 'lon'})

        return data

    aux = xr.open_dataset(pwd_datos + 'pp_20CR-V3.nc')
    pp_20cr = aux.sel(lon=slice(270, 330), lat=slice(-50, 20))

    aux = xr.open_dataset(pwd_datos + 't_20CR-V3.nc')
    t_20cr = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))

    aux = xr.open_dataset(pwd_datos + 't_cru.nc')
    t_cru = ChangeLons(aux)

    ### Precipitation ###
    if name == 'pp_20CR-V3':
        # NOAA20CR-V3
        aux = xr.open_dataset(pwd_datos + 'pp_20CR-V3.nc')
        pp_20cr = aux.sel(lon=slice(270, 330), lat=slice(-50, 20))
        pp_20cr = pp_20cr.rename({'prate': 'var'})
        pp_20cr = pp_20cr.__mul__(86400 * (365 / 12))  # kg/m2/s -> mm/month
        pp_20cr = pp_20cr.drop('time_bnds')

        return pp_20cr
    elif name == 'pp_gpcc':
        # GPCC2018
        aux = xr.open_dataset(pwd_datos + 'pp_gpcc.nc')
        # interpolado igual que 20cr, los dos son 1x1 pero con distinta grilla
        pp_gpcc = aux.sel(lon=slice(270, 330), lat=slice(20, -50))
        if interp:
            pp_gpcc = aux.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
        pp_gpcc = pp_gpcc.rename({'precip': 'var'})

        return pp_gpcc
    elif name == 'pp_PREC':
        # PREC
        aux = xr.open_dataset(pwd_datos + 'pp_PREC.nc')
        pp_prec = aux.sel(lon=slice(270, 330), lat=slice(20, -60))
        if interp:
            pp_prec = pp_prec.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
        pp_prec = pp_prec.rename({'precip': 'var'})
        pp_prec = pp_prec.__mul__(365 / 12)  # mm/day -> mm/month

        return pp_prec
    elif name == 'pp_chirps':
        # CHIRPS
        aux = xr.open_dataset(pwd_datos + 'pp_chirps.nc')
        aux = ChangeLons(aux, 'longitude')
        aux = aux.rename({'precip': 'var', 'latitude': 'lat'})
        aux = aux.sel(lon=slice(270, 330), lat=slice(-50, 20))
        if interp:
            aux = aux.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
        pp_ch = aux

        return pp_ch
    elif name == 'pp_CMAP':
        # CMAP
        aux = xr.open_dataset(pwd_datos + 'pp_CMAP.nc')
        aux = aux.rename({'precip': 'var'})
        aux = aux.sel(lon=slice(270, 330), lat=slice(20, -50))
        if interp:
            pp_cmap = aux.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
        pp_cmap = aux.__mul__(365 / 12)  # mm/day -> mm/month

        return pp_cmap
    elif name == 'pp_gpcp':
        # GPCP2.3
        aux = xr.open_dataset(pwd_datos + 'pp_gpcp.nc')
        aux = aux.rename({'precip': 'var'})
        aux = aux.sel(lon=slice(270, 330), lat=slice(-50, 20))
        if interp:
            pp_gpcp = aux.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
        aux = aux.drop('lat_bnds')
        aux = aux.drop('lon_bnds')
        aux = aux.drop('time_bnds')
        pp_gpcp = aux.__mul__(365 / 12)  # mm/day -> mm/month

        return pp_gpcp
    elif name == 't_20CR-V3':
        # 20CR-v3
        aux = xr.open_dataset(pwd_datos + 't_20CR-V3.nc')
        t_20cr = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))
        t_20cr = t_20cr.rename({'air': 'var'})
        t_20cr = t_20cr - 273
        t_20cr = t_20cr.drop('time_bnds')

        return t_20cr

    elif name == 't_cru':
        # CRU
        aux = xr.open_dataset(pwd_datos + 't_cru.nc')
        t_cru = ChangeLons(aux)
        t_cru = t_cru.sel(lon=slice(270, 330), lat=slice(-60, 20),
                          time=slice('1920-01-01', '2020-12-31'))
        # interpolado a 1x1
        if interp:
            t_cru = t_cru.interp(lat=t_20cr.lat.values, lon=t_20cr.lon.values)
        t_cru = t_cru.rename({'tmp': 'var'})
        t_cru = t_cru.drop('stn')

        return t_cru
    elif name == 't_BEIC': # que mierda pasaAAA!
        # Berkeley Earth etc
        aux = xr.open_dataset(pwd_datos + 't_BEIC.nc')
        aux = aux.rename({'longitude': 'lon', 'latitude': 'lat', 'temperature': 'var'})
        aux = ChangeLons(aux)
        aux = aux.sel(lon=slice(270, 330), lat=slice(-60, 20), time=slice(1920, 2020.999))
        if interp:
            aux = aux.interp(lat=t_20cr.lat.values, lon=t_20cr.lon.values)

        t_cru = t_cru.sel(time=slice('1920-01-01', '2020-12-31'))
        aux['time'] = t_cru.time.values
        aux['month_number'] = t_cru.time.values[-12:]
        t_beic_clim_months = aux.climatology
        t_beic = aux['var']
        # reconstruyendo?¿
        t_beic = t_beic.groupby('time.month') + t_beic_clim_months.groupby('month_number.month').mean()
        t_beic = t_beic.drop('month')
        t_beic = xr.Dataset(data_vars={'var': t_beic})

        return t_beic

    elif name == 't_ghcn_cams':
        # GHCN

        aux = xr.open_dataset(pwd_datos + 't_ghcn_cams.nc')
        aux = aux.sel(lon=slice(270, 330), lat=slice(20, -60))
        if interp:
            aux = aux.interp(lon=t_20cr.lon.values, lat=t_20cr.lat.values)
        t_ghcn = aux.rename({'air': 'var'})
        t_ghcn = t_ghcn - 273

        return t_ghcn

    elif name == 't_hadcrut':
        # HadCRUT
        aux = xr.open_dataset(pwd_datos + 't_hadcrut_anom.nc')
        aux = ChangeLons(aux, 'longitude')
        aux = aux.sel(lon=slice(270, 330), latitude=slice(-60, 20))
        if interp:
            aux = aux.interp(lon=t_20cr.lon.values, latitude=t_20cr.lat.values)
        aux = aux.rename({'tas_mean': 'var', 'latitude': 'lat'})
        t_had = aux.sel(time=slice('1920-01-01', '2020-12-31'))

        aux = xr.open_dataset(pwd_datos + 't_hadcrut_mean.nc')
        aux = ChangeLons(aux)
        aux = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))
        if interp:
            aux = aux.interp(lon=t_20cr.lon.values, lat=t_20cr.lat.values)
        t_had_clim = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))
        aux = aux.rename({'tem': 'var'})
        aux['time'] = t_cru.time.values[-12:]
        # reconstruyendo?¿
        t_had = t_had.groupby('time.month') + aux.groupby('time.month').mean()
        t_had = t_had.drop('realization')
        t_had = t_had.drop('month')

        return t_had

    elif name == 't_era20c':

        # ERA-20C
        aux = xr.open_dataset(pwd_datos + 't_era20c.nc')
        aux = aux.rename({'t2m': 'var', 'latitude': 'lat', 'longitude': 'lon'})
        aux = aux.sel(lon=slice(270, 330), lat=slice(20, -60))
        if interp:
            aux = aux.interp(lon=t_20cr.lon.values, lat=t_20cr.lat.values)
        t_era20 = aux - 273

        return t_era20


os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

w_dir = '/home/luciano.andrian/doc/salidas/'
out_dir = '/home/luciano.andrian/doc/salidas/ENSO_IOD/composite/'
pwd = '/datos/luciano.andrian/ncfiles/'



########################################################################################################################
variables = 'hgt200'
variables_t_pp = ['pp_20CR-V3', 'pp_gpcc', 'pp_PREC', 'pp_chirps', 'pp_CMAP',
                  'pp_gpcp', 't_20CR-V3', 't_cru', 't_BEIC', 't_ghcn_cams','t_hadcrut', 't_era20c']
interp = [False, False, False, False, False, False, False, False, False, False, False, False]

seasons = [7, 9, 10]
seasons_name=['JJA', 'JAS','ASO', 'SON']

two_variables = True
contour = True
SA = True
step = 1
contour0 = False


scales_pp_t = [np.linspace(-30,30,13), # pp
               np.linspace(-1,1,21)] # t
scale = np.linspace(-450, 450, 13)

from matplotlib import colors
cbar_r = colors.ListedColormap(['#B9391B', '#CD4838', '#E25E55', '#F28C89', '#FFCECC',
                              'white',
                              '#B3DBFF', '#83B9EB', '#5E9AD7', '#3C7DC3', '#2064AF'])
cbar_r.set_under('#9B1C00')
cbar_r.set_over('#014A9B')
cbar_r.set_bad(color='white')

cbar = colors.ListedColormap(['#B9391B', '#CD4838', '#E25E55', '#F28C89', '#FFCECC',
                              'white',
                              '#B3DBFF', '#83B9EB', '#5E9AD7', '#3C7DC3', '#2064AF'][::-1])
cbar.set_over('#9B1C00')
cbar.set_under('#014A9B')
cbar.set_bad(color='white')

cmap = ['BrBG', cbar]


save = True
full_season = True
m = 9
season = 9
bwa = False
SA_map = True

########################################################################################################################
for v in range(0,len(variables_t_pp)):

    if v < 6:
        print('pp')
        start = 1985
        end = 2010
        v2 = 0
    else:
        print('temp')
        start = 1950
        end = 2010
        v2 = 1

    dmi = DMI(filter_bwa=False, start_per=start, end_per=end)[0]
    aux = xr.open_dataset("/pikachu/datos4/Obs/sst/sst.mnmean_2020.nc")
    n34 = Nino34CPC(aux, start=start, end=end)[2]


    data_sa = OpenDatasets(name=variables_t_pp[v], interp=interp[v])
    print('data_sa open:' + variables_t_pp[v] + '.nc')
    data_sa = data_sa.sel(time=slice(str(start) + '-01-01', str(end) + '-12-31'))

    data = xr.open_dataset(pwd + variables + '.nc')
    print('data open:' + variables + '.nc')
    data = data.sel(time=slice(str(start) + '-01-01', str(end) + '-12-01'))


    DMI_sim, DMI_un, N34_un, dmi_comp, N34_comp, neutral, DMI_sim_pos, \
    DMI_sim_neg, DMI_un_pos, DMI_un_neg, N34_un_pos, N34_un_neg, DMI_pos, \
    DMI_neg, N34_pos, N34_neg = MultipleComposite(data_sa, n34, dmi, season, start=start,
                                                  full_season=full_season, compute_composite=True)

    DMI_sim2, DMI_un2, N34_un2, dmi_comp2, N34_comp2, neutral2, DMI_sim_pos2, \
    DMI_sim_neg2, DMI_un_pos2, DMI_un_neg2, N34_un_pos2, N34_un_neg2, DMI_pos2, \
    DMI_neg2, N34_pos2, N34_neg2 = MultipleComposite(data, n34, dmi, season, start=start,
                                                     full_season=full_season, compute_composite=True)

    # Sim
    Plots(data=DMI_sim, neutral=neutral, variable='var',
          data2=DMI_sim2, neutral2=neutral2,
          DMI_pos=DMI_sim_pos, DMI_neg=DMI_sim_neg,
          N34_pos=DMI_sim_pos, N34_neg=DMI_sim_neg,
          mode='Simultaneus IODs-ENSOs',
          title=variables_t_pp[v] + ' - ',
          neutral_name='All_Neutral',
          levels=scales_pp_t[v2],
          name_fig= out_dir + 'SIM_' + variables_t_pp[v] + '_SA_',
          save=save, contour=False, waf=False,
          two_variables=True, levels2=scale,
          season='JASON', SA=SA_map,
          step=1, contour0=True,
          cmap=cmap[v2])

    # un
    Plots(data=DMI_un, neutral=neutral, variable='var',
          data2=DMI_un2, neutral2=neutral2,
          DMI_pos=DMI_un_pos, DMI_neg=DMI_un_neg,
          N34_pos=N34_un_pos, N34_neg=N34_un_neg,
          mode='Isolated IODs',
          title=variables_t_pp[v] + ' - ',
          neutral_name='All_Neutral',
          levels=scales_pp_t[v2],
          name_fig= out_dir + 'UN_DMI_' + variables_t_pp[v]+ '_SA_',
          save=save, contour=False, waf=False,
          two_variables=True, levels2=scale,
          season='JASON', SA=SA_map,
          step=1, contour0=True,
          cmap=cmap[v2])

    Plots(data=N34_un, neutral=neutral, variable='var',
          data2=N34_un2, neutral2=neutral2,
          DMI_pos=DMI_un_pos, DMI_neg=DMI_un_neg,
          N34_pos=N34_un_pos, N34_neg=N34_un_neg,
          mode='Isolated ENSOs',
          title=variables_t_pp[v] + ' - ',
          neutral_name='All_Neutral',
          levels=scales_pp_t[v2],
          name_fig= out_dir + 'UN_N34_' + variables_t_pp[v]+ '_SA_',
          save=save, contour=False, waf=False,
          two_variables=True, levels2=scale,
          season='JASON', SA=SA_map,
          step=1, contour0=True,
          cmap=cmap[v2], text=False)

    Plots(data=dmi_comp, neutral=neutral, variable='var',
          data2=dmi_comp2, neutral2=neutral2,
          DMI_pos=DMI_pos, DMI_neg=DMI_neg,
          N34_pos=N34_pos, N34_neg=N34_neg,
          mode='All IODs',
          title=variables_t_pp[v] + ' - ',
          neutral_name='All_Neutral',
          levels=scales_pp_t[v2],
          name_fig= out_dir + 'All_DMI_' + variables_t_pp[v]+ '_SA_',
          save=save, contour=False, waf=False,
          two_variables=True, levels2=scale,
          season='JASON', SA=SA_map,
          step=1, contour0=True,
          cmap=cmap[v2])

    Plots(data=N34_comp, neutral=neutral, variable='var',
          data2=N34_comp2, neutral2=neutral2,
          DMI_pos=DMI_pos, DMI_neg=DMI_neg,
          N34_pos=N34_pos, N34_neg=N34_neg,
          mode='All ENSOs',
          title=variables_t_pp[v] + ' - ',
          neutral_name='All_Neutral',
          levels=scales_pp_t[v2],
          name_fig= out_dir + 'All_N34' + variables_t_pp[v]+ '_SA_',
          save=save, contour=False, waf=False,
          two_variables=True, levels2=scale,
          season='JASON', SA=SA_map,
          step=1, contour0=True,
          cmap=cmap[v2], text=False)




