"""
ENSO vs IOD Regression temporal
"""
from itertools import groupby
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
import xarray as xr
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import statsmodels.formula.api as sm
import os
from ENSO_IOD_Funciones import Nino34CPC
from ENSO_IOD_Funciones import DMI
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

w_dir = '/home/luciano.andrian/doc/salidas/'
out_dir = '/home/luciano.andrian/doc/salidas/ENSO_IOD/regression/Full/'
file_dir = '/datos/luciano.andrian/ncfiles/'
pwd = '/datos/luciano.andrian/ncfiles/'

################################ Functions #############################################################################

def LinearReg(xrda, dim, deg=1):
    # liner reg along a single dimension
    aux = xrda.polyfit(dim=dim, deg=deg, skipna=True)
    return aux

def LinearReg1_D(dmi, n34):
    import statsmodels.formula.api as smf

    df = pd.DataFrame({'dmi': dmi.values, 'n34': n34.values})

    result = smf.ols(formula='n34~dmi', data=df).fit()
    n34_pred_dmi = result.params[1] * dmi.values + result.params[0]

    result = smf.ols(formula='dmi~n34', data=df).fit()
    dmi_pred_n34 = result.params[1] * n34.values + result.params[0]

    return n34 - n34_pred_dmi, dmi - dmi_pred_n34

def is_months(month, mmin, mmax):
    return (month >= mmin) & (month <= mmax)

def RegWEffect(n34, dmi,data=None, data2=None, m=9,two_variables=False):
    var_reg_n34_2=0
    var_reg_dmi_2=1

    data['time'] = n34
     #print('Full Season')
    aux = LinearReg(data.groupby('month')[m], 'time')
    # aux = xr.polyval(data.groupby('month')[m].time, aux.var_polyfit_coefficients[0]) + \
    #       aux.var_polyfit_coefficients[1]
    var_reg_n34 = aux.var_polyfit_coefficients[0]

    data['time'] = dmi
    aux = LinearReg(data.groupby('month')[m], 'time')
    var_reg_dmi = aux.var_polyfit_coefficients[0]

    if two_variables:
        print('Two Variables')

        data2['time'] = n34
        #print('Full Season data2, m ignored')
        aux = LinearReg(data2.groupby('month')[m], 'time')
        var_reg_n34_2 = aux.var_polyfit_coefficients[0]

        data['time'] = dmi
        aux = LinearReg(data2.groupby('month')[m], 'time')
        var_reg_dmi_2 = aux.var_polyfit_coefficients[0]

    return var_reg_n34, var_reg_dmi, var_reg_n34_2, var_reg_dmi_2


def RegWOEffect(n34, n34_wo_dmi, dmi, dmi_wo_n34, m=9, datos=None):

    datos['time'] = n34

    aux = LinearReg(datos.groupby('month')[m], 'time')
    aux = xr.polyval(datos.groupby('month')[m].time, aux.var_polyfit_coefficients[0]) +\
          aux.var_polyfit_coefficients[1]

    #wo n34
    var_regdmi_won34 = datos.groupby('month')[m]-aux

    var_regdmi_won34['time'] = dmi_wo_n34.groupby('time.month')[m] #index wo influence
    var_dmi_won34 = LinearReg(var_regdmi_won34,'time')

    #-----------------------------------------#

    datos['time'] = dmi
    aux = LinearReg(datos.groupby('month')[m], 'time')
    aux = xr.polyval(datos.groupby('month')[m].time, aux.var_polyfit_coefficients[0]) + \
          aux.var_polyfit_coefficients[1]

    #wo dmi
    var_regn34_wodmi = datos.groupby('month')[m]-aux

    var_regn34_wodmi['time'] = n34_wo_dmi.groupby('time.month')[m] #index wo influence
    var_n34_wodmi = LinearReg(var_regn34_wodmi,'time')

    return var_n34_wodmi.var_polyfit_coefficients[0],\
           var_dmi_won34.var_polyfit_coefficients[0],\
           var_regn34_wodmi,var_regdmi_won34

def Corr(datos, index, time_original, m=9):
    aux_corr1 = xr.DataArray(datos.groupby('month')[m]['var'],
                             coords={'time': time_original.groupby('time.month')[m].values,
                                     'lon': datos.lon.values, 'lat': datos.lat.values},
                             dims=['time', 'lat', 'lon'])
    aux_corr2 = xr.DataArray(index.groupby('time.month')[m],
                             coords={'time': time_original.groupby('time.month')[m]},
                             dims={'time'})

    return xr.corr(aux_corr1, aux_corr2, 'time')

def PlotReg(data, data_cor, levels=np.linspace(-100,100,2), cmap='RdBu_r'
            , dpi=100, save=False, title='\m/', name_fig='fig_PlotReg', sig=True
            ,two_variables = False, data2=None, data_cor2=None, levels2 = np.linspace(-100,100,2)
            , sig2=True, step=1,SA=False, contour0=False, color_map = '#d9d9d9'):


    if SA:
        fig = plt.figure(figsize=(5, 6), dpi=dpi)
    else:
        fig = plt.figure(figsize=(7, 3.5), dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    crs_latlon = ccrs.PlateCarree()
    if SA:
        ax.set_extent([270,330, -60,20], crs=crs_latlon)
    else:
        ax.set_extent([0, 359, -80, 40], crs=crs_latlon)



    im = ax.contourf(data.lon[::step], data.lat[::step], data[::step,::step],levels=levels,
                     transform=crs_latlon, cmap=cmap, extend='both')
    if sig:
        ax.contour(data_cor.lon[::step], data_cor.lat[::step], data_cor[::step,::step], levels=np.linspace(-r_crit, r_crit, 2),
                   colors='magenta', transform=crs_latlon, linewidths=1)

    if contour0:
        ax.contour(data.lon, data.lat, data, levels=0,
                   colors='k', transform=crs_latlon, linewidths=1)


    if two_variables:
        ax.contour(data2.lon, data2.lat, data2, levels=levels2,
                   colors='k', transform=crs_latlon, linewidths=1)
        if sig2:
            ax.contour(data_cor2.lon, data_cor2.lat, data_cor2, levels=np.linspace(-r_crit, r_crit, 2),
                       colors='forestgreen', transform=crs_latlon, linewidths=1)

    cb = plt.colorbar(im, fraction=0.042, pad=0.035,shrink=0.8)
    cb.ax.tick_params(labelsize=8)
    ax.add_feature(cartopy.feature.LAND, facecolor=color_map)
    ax.add_feature(cartopy.feature.COASTLINE)
    # ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
    if SA:
        ax.set_xticks(np.arange(270, 330, 10), crs=crs_latlon)
        ax.set_yticks(np.arange(-60, 20, 20), crs=crs_latlon)
    else:
        ax.set_xticks(np.arange(30, 330, 60), crs=crs_latlon)
        ax.set_yticks(np.arange(-80, 40, 20), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=7)
    plt.title(title, fontsize=10)
    plt.tight_layout()

    if save:
        print('save: ' + out_dir + name_fig + '.jpg')
        plt.savefig(out_dir + name_fig + '.jpg')
        plt.close()

    else:
        plt.show()

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
    pp_20cr = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))

    aux = xr.open_dataset(pwd_datos + 't_20CR-V3.nc')
    t_20cr = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))

    aux = xr.open_dataset(pwd_datos + 't_cru.nc')
    t_cru = ChangeLons(aux)

    ### Precipitation ###
    if name == 'pp_20CR-V3':
        # NOAA20CR-V3
        aux = xr.open_dataset(pwd_datos + 'pp_20CR-V3.nc')
        pp_20cr = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))
        pp_20cr = pp_20cr.rename({'prate': 'var'})
        pp_20cr = pp_20cr.__mul__(86400 * (365 / 12))  # kg/m2/s -> mm/month
        pp_20cr = pp_20cr.drop('time_bnds')

        return pp_20cr
    elif name == 'pp_gpcc':
        # GPCC2018
        aux = xr.open_dataset(pwd_datos + 'pp_gpcc.nc')
        # interpolado igual que 20cr, los dos son 1x1 pero con distinta grilla
        pp_gpcc = aux.sel(lon=slice(270, 330), lat=slice(20, -60))
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
        aux = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))
        if interp:
            aux = aux.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
        pp_ch = aux

        return pp_ch
    elif name == 'pp_CMAP':
        # CMAP
        aux = xr.open_dataset(pwd_datos + 'pp_CMAP.nc')
        aux = aux.rename({'precip': 'var'})
        aux = aux.sel(lon=slice(270, 330), lat=slice(20, -60))
        if interp:
            pp_cmap = aux.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
        pp_cmap = aux.__mul__(365 / 12)  # mm/day -> mm/month

        return pp_cmap
    elif name == 'pp_gpcp':
        # GPCP2.3
        aux = xr.open_dataset(pwd_datos + 'pp_gpcp.nc')
        aux = aux.rename({'precip': 'var'})
        aux = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))
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
    elif name == 'pp_lieb':
        aux = xr.open_dataset(pwd_datos + 'pp_liebmann.nc')
        aux = aux.sel(time=slice('1985-01-01', '2010-12-31'))
        aux = aux.resample(time='1M', skipna=True).mean()
        aux = ChangeLons(aux, 'lon')
        pp_lieb = aux.sel(lon=slice(275, 330), lat=slice(-60, 20))
        pp_lieb = pp_lieb.__mul__(365 / 12)
        pp_lieb = pp_lieb.drop('count')
        pp_lieb = pp_lieb.rename({'precip': 'var'})
        return pp_lieb


def ComputeWithEffect(data=None, data2=None, n34=None, dmi=None,
                     two_variables=False, full_season=False,
                     time_original=None,m=9):
    print('Reg...')
    print('#-- With influence --#')
    aux_n34, aux_dmi, aux_n34_2, aux_dmi_2 = RegWEffect(data=data, data2=data2,
                                                       n34=n34.__mul__(1 / n34.std('time')),
                                                       dmi=dmi.__mul__(1 / dmi.std('time')),
                                                       m=m, two_variables=two_variables)
    if full_season:
        print('Full Season')
        n34 = n34.rolling(time=5, center=True).mean()
        dmi = dmi.rolling(time=5, center=True).mean()

    print('Corr...')
    aux_corr_n34 = Corr(datos=data, index=n34, time_original=time_original, m=m)
    aux_corr_dmi = Corr(datos=data, index=dmi, time_original=time_original, m=m)

    aux_corr_dmi_2 = 0
    aux_corr_n34_2 = 0
    if two_variables:
        print('Corr2..')
        aux_corr_n34_2 = Corr(datos=data2, index=n34, time_original=time_original, m=m)
        aux_corr_dmi_2 = Corr(datos=data2, index=dmi, time_original=time_original, m=m)

    return aux_n34, aux_corr_n34, aux_dmi, aux_corr_dmi, aux_n34_2, aux_corr_n34_2, aux_dmi_2, aux_corr_dmi_2


def ComputeWithoutEffect(data, n34, dmi, m):
    # -- Without influence --#
    print('# -- Without influence --#')
    print('Reg...')
    # dmi wo n34 influence and n34 wo dmi influence
    dmi_wo_n34, n34_wo_dmi = LinearReg1_D(n34.__mul__(1 / n34.std('time')),
                                          dmi.__mul__(1 / dmi.std('time')))

    # Reg WO
    aux_n34_wodmi, aux_dmi_won34, data_n34_wodmi, data_dmi_won34 = \
        RegWOEffect(n34=n34.__mul__(1 / n34.std('time')),
                   n34_wo_dmi=n34_wo_dmi,
                   dmi=dmi.__mul__(1 / dmi.std('time')),
                   dmi_wo_n34=dmi_wo_n34,
                   m=m, datos=data)

    print('Corr...')
    aux_corr_n34 = Corr(datos=data_n34_wodmi, index=n34_wo_dmi, time_original=time_original,m=m)
    aux_corr_dmi = Corr(datos=data_dmi_won34, index=dmi_wo_n34, time_original=time_original,m=m)

    return aux_n34_wodmi, aux_corr_n34, aux_dmi_won34, aux_corr_dmi

# ########################################################################################################################
#----------------------------------------------------------------------#

variables = ['psl','pp_gpcc','t_cru', 't_BEIC','hgt200','sf', 'div', 'vp']
interp = [False, False, False, False, False, False, False, False, False, False, False, False]
seasons = [7, 8, 9, 10] # main month

var_name = ['psl','var', 'var', 'var', 'z','streamfunction', 'divergence','velocity_potential']
title_var = ['PSL', 'PP','Temp-CRU', 'Temp-BEIC', 'HGT', 'Psi', 'Divergence', 'Potential Velocity']

two_variables = [False, True, True, True, False,False, True, False]
SA = [False, True,  True, True, False, False, False, False]
step = [1,1,1,1,1,1,10,1]
sig = [True, True, True, True, True, True, False, True]

scales = [np.linspace(-1.2,1.2,13),  #psl
          np.linspace(-15, 15, 13),  # pp
          np.linspace(-0.8, 0.8, 17),  # t
          np.linspace(-0.8, 0.8, 17),  # t
          np.linspace(-150, 150, 13),  #hgt
          np.linspace(-2.4e6,2.4e6,13),  #sf
          np.linspace(-0.21e-5,0.21e-5,13),  #div
          np.linspace(-2.5e6,2.5e6,13)]#vp

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

cmap = [cbar,'BrBG',cbar,cbar,cbar,cbar_r,cbar,cbar]

save = True
full_season = False
text = False
# m = 9
#start = [1920,1950]
i=1920
end = 2020

#----------------------------------------------------------------------------------------------------------------------#

########################################################################################################################


count=0
for v in variables:
    #for i in start:
    end = 2020
        # 1920-2020 t=1.660
        # 1950-2020 t=1.667
    t = 1.66
    r_crit = np.sqrt(1 / (((np.sqrt((end - i) - 2) / t) ** 2) + 1))

    # indices: ----------------------------------------------------------------------------------------------------#
    dmi = DMI(filter_bwa=False, start_per=str(i), end_per=str(end))[2]
    dmi = dmi.sel(time=slice(str(i) + '-01-01', str(end) + '-12-01'))
    aux = xr.open_dataset("/pikachu/datos4/Obs/sst/sst.mnmean_2020.nc")
    n34 = Nino34CPC(aux, start=i)[0]
    n34 = n34.sel(time=slice(str(i) + '-01-01', str(end) + '-12-01'))


    # Open: -------------------------------------------------------------------------------------------------------#
    if (v != variables[1]) & (v != variables[2]) & (v != variables[3]):
        data = xr.open_dataset(file_dir + v + '.nc')
        print('data open:' + v + '.nc')
        data = data.sel(time=slice(str(i) + '-01-01', str(end) + '-12-01'))
        time_original = data.time
    else:
        print('Using OpenDatasets')
        data = OpenDatasets(name=v, interp=False)
        print('data_sa open:' + v + '.nc')
        data = data.sel(time=slice(str(i) + '-01-01', str(end) + '-12-31'))
        time_original = data.time

    end_or = end
    end = int(time_original[-1].dt.year.values)
    print(str(end) + ' -  end year from main data')



    if v == variables[0]:
        data = data.__mul__(1 / 100)
    elif v == variables[4]:
        data = data.drop('level')


    # Anomaly -----------------------------------------------------------------------------------------------------#
    data = data.groupby('time.month') - data.groupby('time.month').mean('time', skipna=True)

    data2 = None
    scale2 = None
    if two_variables[count]:
        print('Two Variables')
        if (v == variables[1]) | (v == variables[2]) | (v == variables[3]):
            v2 = 'psl'
            scale2 = scales[0]
        elif v == variables[-2]:
            v2 = 'vp'
            scale2 = scales[-1]


        data2 = xr.open_dataset(file_dir + v2 + '1x1.nc') # es para contornos.. -ram
        print('data open:' + v2 + '.nc')
        data2 = data2.sel(time=slice(str(i) + '-01-01', str(end) + '-12-01'))

        if len(data2.sel(lat=slice(-60, 20)).lat.values) == 0:
            data2 = data2.sel(lat=slice(20, -60))
        else:
            data2 = data2.sel(lat=slice(-60, 20))

            time_original = data2.time

        data2 = data2.groupby('time.month') - data2.groupby('time.month').mean('time', skipna=True)

        if v2 == 'psl':
            data2 = data2.__mul__(1 / 100)

        if end != end_or:
            n34 = n34.sel(time=slice('1920-01-01', str(end) + '-12-01'))
            dmi = dmi.sel(time=slice('1920-01-01', str(end) + '-12-01'))

    # 3/5-month running mean --------------------------------------------------------------------------------------#
    if full_season:
        print('FULL SEASON JASON')
        text = False
        m = 9
        print('full season rolling')
        seasons_name = 'JASON'
        data = data.rolling(time=5, center=True).mean()
        if two_variables[count]:
            data2 = data2.rolling(time=5, center=True).mean()

        aux_n34, aux_corr_n34, aux_dmi, \
        aux_corr_dmi, aux_n34_2, aux_corr_n34_2,\
        aux_dmi_2, aux_corr_dmi_2 = ComputeWithEffect(data=data, data2=data2, n34=n34, dmi=dmi,
                                                      two_variables=two_variables[count],m=m,
                                                      full_season=full_season, time_original=time_original)

        print('Plot...')
        PlotReg(data=aux_n34, data_cor=aux_corr_n34,
                levels=scales[count], cmap=cmap[count], dpi=200,
                title=title_var[count] + '_' + seasons_name +
                      '_' + str(i) + '_' + str(end) + '_Niño3.4',
                name_fig=v + '_' + seasons_name + str(i) + '_' + str(end) + '_N34',
                save=save, sig=True,
                two_variables=two_variables[count],
                data2=aux_n34_2, data_cor2=aux_corr_n34_2,
                levels2=scale2, sig2=True,
                SA=SA[count], step=step[count], contour0=False, color_map='k')

        PlotReg(data=aux_dmi, data_cor=aux_corr_dmi,
                levels=scales[count], cmap=cmap[count], dpi=200,
                title=title_var[count] + '_' + seasons_name +
                      '_' + str(i) + '_' + str(end) + '_DMI',
                name_fig=v + '_' + seasons_name + str(i) + '_' + str(end) + '_DMI',
                save=save, sig=True,
                two_variables=two_variables[count],
                data2=aux_dmi_2, data_cor2=aux_corr_dmi_2,
                levels2=scales[count], sig2=True,
                SA=SA[count], step=step[count], contour0=False, color_map='k')

        del aux_n34, aux_dmi, aux_n34_2, aux_dmi_2, aux_corr_dmi, aux_corr_n34, \
                aux_corr_dmi_2, aux_corr_n34_2


        aux_n34_wodmi, aux_corr_n34, aux_dmi_won34, aux_corr_dmi = ComputeWithoutEffect(data, n34, dmi, m)

        aux_n34_wodmi_2 = 0
        aux_corr_n34_2 = 0
        aux_dmi_won34_2 = 0
        aux_corr_dmi_2 = 0


        if two_variables[count]:
            aux_n34_wodmi_2, aux_corr_n34_2, \
            aux_dmi_won34_2, aux_corr_dmi_2 = ComputeWithoutEffect(data2, n34, dmi, m)

        print('Plot...')
        PlotReg(data=aux_n34_wodmi, data_cor=aux_corr_n34,
                levels=scales[count], cmap=cmap[count], dpi=200,
                title=title_var[count] + '_' + seasons_name +
                      '_' + str(i) + '_' + str(end) + '_Niño3.4 -{DMI}',
                name_fig=v + '_' + seasons_name + str(i) + '_' + str(end) + '_N34_wodmi',
                save=save, sig=True,
                two_variables=two_variables[count],
                data2=aux_n34_wodmi_2, data_cor2=aux_corr_n34_2,
                levels2=scale2, sig2=True,
                SA=SA[count], step=step[count], contour0=False, color_map='k')


        PlotReg(data=aux_dmi_won34, data_cor=aux_corr_dmi,
                levels=scales[count], cmap=cmap[count], dpi=200,
                title=title_var[count] + '_' + seasons_name +
                      '_' + str(i) + '_' + str(end) + '_DMI -{N34}',
                name_fig=v + '_' + seasons_name + str(i) + '_' + str(end) + '_DMI_woN34',
                save=save, sig=True,
                two_variables=two_variables[count],
                data2=aux_dmi_won34_2, data_cor2=aux_corr_dmi_2,
                levels2=scale2, sig2=True,
                SA=SA[count], step=step[count], contour0=False, color_map='k')



        del aux_n34_wodmi, aux_dmi_won34, aux_corr_dmi, aux_corr_n34,\
            aux_n34_wodmi_2, aux_dmi_won34_2, aux_corr_dmi_2, aux_corr_n34_2
        ################################################################################################################
        ################################################################################################################
    else:
        seasons_name = ['JJA', 'JAS', 'ASO', 'SON']

        print('season rolling')
        data = data.rolling(time=3, center=True).mean()
        if two_variables[count]:
            data2 = data2.rolling(time=3, center=True).mean()

        count_season = 0
        for m in seasons:

            print(seasons_name[m - 7])
            print(m)
            aux_n34, aux_corr_n34, aux_dmi, \
            aux_corr_dmi, aux_n34_2, aux_corr_n34_2, \
            aux_dmi_2, aux_corr_dmi_2 = ComputeWithEffect(data=data, data2=data2, n34=n34, dmi=dmi,
                                                          two_variables=two_variables[count], m=m,
                                                          full_season=False, time_original=time_original)

            print('Plot')
            PlotReg(data=aux_n34, data_cor=aux_corr_n34,
                    levels=scales[count], cmap=cmap[count], dpi=200,
                    title=title_var[count] + '_' + seasons_name[count_season] +
                          '_' + str(i) + '_' + str(end) + '_Niño3.4',
                    name_fig=v + '_' + seasons_name[count_season] + '_' + str(i) +
                             '_' + str(end) + '_N34',
                    save=save, sig=True,
                    two_variables=two_variables[count],
                    data2=aux_n34_2, data_cor2=aux_corr_n34_2,
                    levels2=scale2, sig2=True,
                    SA=SA[count], step=step[count], contour0=False, color_map='k')

            PlotReg(data=aux_dmi, data_cor=aux_corr_dmi,
                    levels=scales[count], cmap=cmap[count], dpi=200,
                    title=title_var[count] + '_' + seasons_name[count_season] +
                          '_' + str(i) + '_' + str(end) + '_DMI',
                    name_fig=v + '_' + seasons_name[count_season] + '_' + str(i) +
                             '_' + str(end) + '_DMI',
                    save=save, sig=True,
                    two_variables=two_variables[count],
                    data2=aux_dmi_2, data_cor2=aux_corr_dmi_2,
                    levels2=scales[count], sig2=True,
                    SA=SA[count], step=step[count], contour0=False, color_map='k')


            del aux_n34, aux_dmi, aux_n34_2, aux_dmi_2, aux_corr_dmi, aux_corr_n34, \
                aux_corr_dmi_2, aux_corr_n34_2

            aux_n34_wodmi, aux_corr_n34, aux_dmi_won34, aux_corr_dmi = ComputeWithoutEffect(data, n34, dmi, m)

            aux_n34_wodmi_2 = 0
            aux_corr_n34_2 = 0
            aux_dmi_won34_2 = 0
            aux_corr_dmi_2 = 0

            if two_variables[count]:
                aux_n34_wodmi_2, aux_corr_n34_2, \
                aux_dmi_won34_2, aux_corr_dmi_2 = ComputeWithoutEffect(data2, n34, dmi, m)

            print('Plot...')
            PlotReg(data=aux_n34_wodmi, data_cor=aux_corr_n34,
                    levels=scales[count], cmap=cmap[count], dpi=200,
                    title=title_var[count] + '_' + seasons_name[count_season] +
                          '_' + str(i) + '_' + str(end) + '_Niño3.4 -{DMI}',
                    name_fig=v + '_' + seasons_name[count_season] + '_' + str(i) +
                             '_' + str(end) + '_N34_woDMI',
                    save=save, sig=True,
                    two_variables=two_variables[count],
                    data2=aux_n34_wodmi_2, data_cor2=aux_corr_n34_2,
                    levels2=scale2, sig2=True,
                    SA=SA[count], step=step[count], contour0=False, color_map='k')

            PlotReg(data=aux_dmi_won34, data_cor=aux_corr_dmi,
                    levels=scales[count], cmap=cmap[count], dpi=200,
                    title=title_var[count] + '_' + seasons_name[count_season] +
                          '_' + str(i) + '_' + str(end) + '_DMI -{N34}',
                    name_fig=v + '_' + seasons_name[count_season] + '_' + str(i) +
                             '_' + str(end) + '_DMI_woN34',
                    save=save, sig=True,
                    two_variables=two_variables[count],
                    data2=aux_dmi_won34_2, data_cor2=aux_corr_dmi_2,
                    levels2=scale2, sig2=True,
                    SA=SA[count], step=step[count], contour0=False, color_map='k')

            del aux_n34_wodmi, aux_dmi_won34, aux_corr_dmi, aux_corr_n34, \
                aux_n34_wodmi_2, aux_dmi_won34_2, aux_corr_dmi_2, aux_corr_n34_2

            count_season += 1
    count += 1