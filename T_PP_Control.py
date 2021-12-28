"""
Control de bases de datos T y PP
"""
import datetime
from itertools import groupby
import xarray as xr
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import statsmodels.formula.api as sm
import os

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import matplotlib.pyplot as plt
# periodos en comun
# PP 1979-2012 TODOS
# sin cmap y gpcp 1948-2012
# sin prec/l 1948-2015 # VER
# solo 20CRv3 y GPCC 1920-2015

# T
#1948-2015 TODOS
# sin GHCN-CAMS 1920-2015
# sin 20CRv· 1920-2020

pwd_datos = '/datos/luciano.andrian/ncfiles/'
pdw_output = '/home/luciano.andrian/doc/salidas/datasets_T_PP/'

########################################################################################################################

def ACC(data1, data2):
    anios = len(data1.time.values)
    aux = data1 * data2
    num = aux.sum('time').__mul__(1 / anios)

    aux2 = (data1 * data1).sum('time') * (data2 * data2).sum('time')
    den = aux2.__mul__(1 / anios)

    return num / np.sqrt(den)

def is_months(month, mmin, mmax):
    return (month >= mmin) & (month <= mmax)

def AnomSeason(data, no_anom=False):
    if no_anom:
        data_3rm = data.rolling(time=3, center=True).mean(skipna=True)
        data_anom = data_3rm.groupby('time.month')
    else:
        data_3rm = data.rolling(time=3, center=True).mean(skipna=True)
        data_anom = data_3rm.groupby('time.month') - data_3rm.groupby('time.month').mean('time', skipna=True)

    return data_anom

def MetricasClim(data1, data2):

    def bias(data1, data2):
        anios = len(data1.time.values)
        aux = data1 - data2
        aux = aux.sum('time').__mul__(1 / anios)
        return aux

    def mae(data1, data2):
        anios = len(data1.time.values)
        aux = np.abs(data1 - data2)
        aux = aux.sum('time').__mul__(1 / anios)
        return aux

    def rmse(data1, data2):
        anios = len(data1.time.values)
        aux = (data1 - data2) ** 2
        aux = np.sqrt(aux.sum('time').__mul__(1 / anios))
        return aux

    bias = bias(data1, data2)
    mae = mae(data1, data2)
    rmse = rmse(data1, data2)
    return bias, mae, rmse

def biastemp(data1, data2, data1_beic=False, data2_beic=False):
    anios = len(data1.time.values)
    if data1_beic:
        aux = (data1.temp /data2.temp.values)
    elif data2_beic:
        aux = (data1.temp.values - data2.temp)
    else:
        aux = (data1.temp.values - data2.temp.values)

    #aux = ((data1.temp.mean('time')/ data2.temp.mean('time'))*100 - 100)
    aux = xr.DataArray(aux, coords=[data1.time.values, data1.lat.values, data1.lon.values],
                       dims=['time', 'lat', 'lon'])
    aux = aux.sum('time').__mul__(1 / anios)
    #aux2 = (aux/data1.temp.mean('time'))*100
    return aux

def maetemp(data1, data2, data1_beic=False, data2_beic=False):
    anios = len(data1.time.values)
    if data1_beic:
        aux = np.abs(data1.temp - data2.temp.values)
    elif data2_beic:
        aux = np.abs(data1.temp.values - data2.temp)
    else:
        aux = np.abs(data1.temp.values - data2.temp.values)

    aux = xr.DataArray(aux, coords=[data1.time.values, data1.lat.values, data1.lon.values],
                       dims=['time', 'lat', 'lon'])
    aux = aux.sum('time').__mul__(1 / anios)
    return aux

def rmsetemp(data1, data2, data1_beic=False, data2_beic=False):
    anios = len(data1.time.values)

    if data1_beic:
        aux = (data1.temp - data2.temp.values) ** 2
    elif data2_beic:
        aux = (data1.temp.values - data2.temp) ** 2
    else:
        aux =(data1.temp.values - data2.temp.values) ** 2
    aux = xr.DataArray(aux, coords=[data1.time.values, data1.lat.values, data1.lon.values],
                       dims=['time', 'lat', 'lon'])
    aux = np.sqrt(aux.sum('time').__mul__(1 / anios))

    return aux

def ClimSeason(data):
    return data.rolling(time=3, center=True).mean(skipna=True)

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

def Plots(data=None, data_var=None,
          levels=np.linspace(0, 1, 11), cmap='Reds',
          contour=False, contour0=False,
          contour_sig=False, rc=0,
          title='', name_fig='fig', dpi=200,
          step=1, save=False,pp=False):

    import cartopy.feature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import cartopy.crs as ccrs

    from numpy import ma
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5, 6), dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    crs_latlon = ccrs.PlateCarree()
    ax.set_extent([270, 330, -60, 20], crs_latlon)

    if pp:
        im = ax.contourf(data.lon[::step], data.lat[::step], data_var.pp[::step, ::step],
                         levels=levels, transform=crs_latlon, cmap=cmap, extend='both')
        if contour:
            values = ax.contour(data.lon, data.lat, data_var.pp, levels=levels,
                                transform=crs_latlon, colors='k', linewidths=1)

        if contour0:
            values = ax.contour(data.lon, data.lat, data_var.pp, levels=0,
                                transform=crs_latlon, colors='gray', linewidths=1.5)
            ax.clabel(values, inline=1.5, fontsize=5, fmt='%1.1f')

        if contour_sig:
            values = ax.contour(data.lon, data.lat, data_var.pp, levels=[-rc, rc],
                                transform=crs_latlon, colors='Green', linewidths=1.5)
    else:
        im = ax.contourf(data.lon[::step], data.lat[::step], data_var[::step, ::step],
                         levels=levels, transform=crs_latlon, cmap=cmap, extend='both')
        if contour:
            values = ax.contour(data.lon, data.lat, data_var, levels=levels,
                                transform=crs_latlon, colors='k', linewidths=1)

        if contour0:
            values = ax.contour(data.lon, data.lat, data_var, levels=0,
                                transform=crs_latlon, colors='gray', linewidths=1.5)
            ax.clabel(values, inline=1.5, fontsize=5, fmt='%1.1f')

        if contour_sig:
            values = ax.contour(data.lon, data.lat, data_var, levels=[-rc, rc],
                                transform=crs_latlon, colors='Green', linewidths=1.5)



    cb = plt.colorbar(im, fraction=0.042, pad=0.035, shrink=0.8)
    cb.ax.tick_params(labelsize=8)
    ax.add_feature(cartopy.feature.LAND, facecolor='#d9d9d9')
    ax.add_feature(cartopy.feature.COASTLINE)
    # ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
    ax.set_xticks(np.arange(270, 330, 10), crs=crs_latlon)
    ax.set_yticks(np.arange(-60, 40, 20), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=7)
    plt.title(title, fontsize=12)
    plt.tight_layout()

    if save:
        plt.savefig(pdw_output + name_fig + '.jpg')
        plt.close()
    else:
        plt.show()

def MultPlot(data_bias=None, data_bias_var=None, levels_bias=np.linspace(-1, 1, 11),
             data_mae=None, data_mae_var=None, levels_mae=np.linspace(-1, 1, 11),
             data_rmse=None, data_rmse_var=None, levels_rmse=np.linspace(-1, 1, 11),
             cmap='RdBu_r', save=False, var='temp',
             title='', name_fig='Fig', s_name=None,pp=False):

    if var == 'temp':
        cbars = ['RdBu_r', 'BuGn', 'CMRmap_r']
        var_title = 'T'
    elif var == 'pp':
        cbars = ['BrBG', 'BuGn', 'CMRmap_r']
        var_title = 'pp'


    seasons = [1, 4, 7, 10]
    season_name = ['DJF', 'MAM', 'JJA', 'SON']


    Plots(data=data_bias, data_var=data_bias_var, levels=levels_bias,
          cmap=cbars[0], save=save,
          title='Bias' + title + str(season_name[s_name]),pp=pp,
          name_fig= var_title + '_BIAS_' + str(season_name[s_name]) + name_fig, contour0=True)

    Plots(data=data_mae, data_var=data_mae_var, levels=levels_mae,
          cmap=cbars[0], save=save,
          title='MAE' + title + str(season_name[s_name]),pp=pp,
          name_fig=var_title + '_mae_' + str(season_name[s_name]) + name_fig)

    Plots(data=data_rmse, data_var=data_rmse_var, levels=levels_rmse,
          cmap=cbars[2], save=save,
          title='RMSE' + title + str(season_name[s_name]),pp=pp,
          name_fig=var_title + '_rmse_' + str(season_name[s_name]) + name_fig)

def PlotsMetricsTemp(start, end, save = False,
                 noaa20c=True, cru=True, beic=True, ghcn=True, had=True, era20=True):
    seasons = [1, 4, 7, 10]
    season_name = ['DJF', 'MAM', 'JJA', 'SON']

    scales = [np.linspace(-4, 4, 17),
              np.linspace(0, 80, 9),
              np.linspace(0, 5, 11)]
    cbars = ['RdBu_r', 'BuGn', 'CMRmap_r']


    # n20cr_anom = AnomSeason(t_20cr.sel(time=slice(start + '-01-01', end + '-12-31')))
    # cru_anom = AnomSeason(t_cru.sel(time=slice(start + '-01-01', end + '-12-31')))
    # beic_anom = AnomSeason(t_beic.sel(time=slice(start + '-01-01', end + '-12-31')))
    # ghcn_anom = AnomSeason(t_ghcn.sel(time=slice(start + '-01-01', end + '-12-31')))
    # had_anom = AnomSeason(t_had.sel(time=slice(start + '-01-01', end + '-12-31')))
    # era20_anom = AnomSeason(t_era20.sel(time=slice(start + '-01-01', end + '-12-31')))

    n20cr_clim = ClimSeason(t_20cr.sel(time=slice(start + '-01-01', end + '-12-31')))
    cru_clim = ClimSeason(t_cru.sel(time=slice(start + '-01-01', end + '-12-31')))
    beic_clim = ClimSeason(t_beic.sel(time=slice(start + '-01-01', end + '-12-31')))
    ghcn_clim = ClimSeason(t_ghcn.sel(time=slice(start + '-01-01', end + '-12-31')))
    had_clim = ClimSeason(t_had.sel(time=slice(start + '-01-01', end + '-12-31')))
    era20_clim = ClimSeason(t_era20.sel(time=slice(start + '-01-01', end + '-12-31')))

    for s in seasons:

        if s == 1:
            s_name = 0
        elif s == 4:
            s_name = 1
        elif s == 7:
            s_name = 2
        elif s == 10:
            s_name = 3

        if noaa20c:
            s_20cr = n20cr_clim.groupby('time.month')[s]

        if cru:
            s_cru = cru_clim.groupby('time.month')[s]

        if beic:
            s_beic = beic_clim.groupby('time.month')[s]


        if ghcn:
            s_ghcn = ghcn_clim.groupby('time.month')[s]

        if had:
            s_had = had_clim.groupby('time.month')[s]

        if era20:
            s_era20 = era20_clim.groupby('time.month')[s]


        if noaa20c & cru & beic & ghcn & had & era20:
            bias1 = biastemp(s_cru, s_20cr)
            bias2 = biastemp(s_cru, s_beic, data2_beic=True)
            bias3 = biastemp(s_cru, s_ghcn)
            bias4 = biastemp(s_cru, s_had)
            bias5 = biastemp(s_cru, s_era20)

            mae1 = maetemp(s_cru, s_20cr)
            mae2 = maetemp(s_cru, s_beic, data2_beic=True)
            mae3 = maetemp(s_cru, s_ghcn)
            mae4 = maetemp(s_cru, s_had)
            mae5 = maetemp(s_cru, s_era20)

            rmse1 = rmsetemp(s_cru, s_20cr)
            rmse2 = rmsetemp(s_cru, s_beic,data2_beic=True)
            rmse3 = rmsetemp(s_cru, s_ghcn)
            rmse4 = rmsetemp(s_cru, s_had)
            rmse5 = rmsetemp(s_cru, s_era20)

            MultPlot(data_bias=bias1, data_bias_var=bias1, levels_bias=scales[0],
                    data_mae=mae1, data_mae_var=mae1, levels_mae=scales[0],
                    data_rmse=rmse1, data_rmse_var=rmse1, levels_rmse=scales[2],
                    title=' CRU - NOAA20C - 1948-2010 - ', name_fig='1_1948_2010',
                    s_name=s_name, save=save)

            MultPlot(data_bias=bias2, data_bias_var=bias2, levels_bias=scales[0],
                    data_mae=mae2, data_mae_var=mae2, levels_mae=scales[0],
                    data_rmse=rmse2, data_rmse_var=rmse2, levels_rmse=scales[2],
                    title=' CRU - BEIC - 1948-2010 - ', name_fig='2_1948_2010',
                    s_name=s_name, save=save)

            MultPlot(data_bias=bias3, data_bias_var=bias3, levels_bias=scales[0],
                    data_mae=mae3, data_mae_var=mae3, levels_mae=scales[0],
                    data_rmse=rmse3, data_rmse_var=rmse3, levels_rmse=scales[2],
                    title=' CRU - GHCN-CAMS - 1948-2010 - ', name_fig='3_1948_2010',
                    s_name=s_name, save=save)

            MultPlot(data_bias=bias4, data_bias_var=bias4, levels_bias=scales[0],
                    data_mae=mae4, data_mae_var=mae4, levels_mae=scales[0],
                    data_rmse=rmse4, data_rmse_var=rmse4, levels_rmse=scales[2],
                    title=' CRU - HADcrut - 1948-2010', name_fig='4_1948_2010',
                    s_name=s_name, save=save)

            MultPlot(data_bias=bias5, data_bias_var=bias5, levels_bias=scales[0],
                    data_mae=mae5, data_mae_var=mae5, levels_mae=scales[0],
                    data_rmse=rmse5, data_rmse_var=rmse5, levels_rmse=scales[2],
                    title=' CRU - ERA20C - 1948-2010', name_fig='5_1948_2010',
                    s_name=s_name, save=save)


        elif noaa20c & cru & beic & had & era20 & (ghcn==False):
            bias1 = biastemp(s_cru, s_20cr)
            bias2 = biastemp(s_cru, s_beic, data2_beic=True)
            bias4 = biastemp(s_cru, s_had)
            bias5 = biastemp(s_cru, s_era20)

            mae1 = maetemp(s_cru, s_20cr)
            mae2 = maetemp(s_cru, s_beic, data2_beic=True)
            mae4 = maetemp(s_cru, s_had)
            mae5 = maetemp(s_cru, s_era20)

            rmse1 = rmsetemp(s_cru, s_20cr)
            rmse2 = rmsetemp(s_cru, s_beic,data2_beic=True)
            rmse4 = rmsetemp(s_cru, s_had)
            rmse5 = rmsetemp(s_cru, s_era20)

            MultPlot(data_bias=bias1, data_bias_var=bias1, levels_bias=scales[0],
                    data_mae=mae1, data_mae_var=mae1, levels_mae=scales[0],
                    data_rmse=rmse1, data_rmse_var=rmse1, levels_rmse=scales[2],
                    title=' CRU - NOAA20C - 1910-2010 - ', name_fig='1_1920_2010',
                    s_name=s_name, save=save)

            MultPlot(data_bias=bias2, data_bias_var=bias2, levels_bias=scales[0],
                    data_mae=mae2, data_mae_var=mae2, levels_mae=scales[0],
                    data_rmse=rmse2, data_rmse_var=rmse2, levels_rmse=scales[2],
                    title=' CRU - BEIC - 1920-2010 - ', name_fig='2_1920_2010',
                    s_name=s_name, save=save)


            MultPlot(data_bias=bias4, data_bias_var=bias4, levels_bias=scales[0],
                    data_mae=mae4, data_mae_var=mae4, levels_mae=scales[0],
                    data_rmse=rmse4, data_rmse_var=rmse4, levels_rmse=scales[2],
                    title=' CRU - HADcrut - 1920-2010 -', name_fig='4_1920_2010',
                    s_name=s_name, save=save)

            MultPlot(data_bias=bias5, data_bias_var=bias5, levels_bias=scales[0],
                    data_mae=mae5, data_mae_var=mae5, levels_mae=scales[0],
                    data_rmse=rmse5, data_rmse_var=rmse5, levels_rmse=scales[2],
                    title=' CRU - ERA20C - 1920-2010 - ', name_fig='5_1920_2010',
                    s_name=s_name, save=save)

        elif noaa20c & cru & beic & had & (era20==False) & (ghcn==False):

            bias2 = biastemp(s_cru, s_beic, data2_beic=True)
            bias4 = biastemp(s_cru, s_had)

            mae2 = maetemp(s_cru, s_beic, data2_beic=True)
            mae4 = maetemp(s_cru, s_had)

            rmse2 = rmsetemp(s_cru, s_beic,data2_beic=True)
            rmse4 = rmsetemp(s_cru, s_had)


            MultPlot(data_bias=bias2, data_bias_var=bias2, levels_bias=scales[0],
                    data_mae=mae2, data_mae_var=mae2, levels_mae=scales[0],
                    data_rmse=rmse2, data_rmse_var=rmse2, levels_rmse=scales[2],
                    title=' CRU - BEIC - 1920-2020 - ', name_fig='2_1920_2020',
                    s_name=s_name, save=save)

            MultPlot(data_bias=bias4, data_bias_var=bias4, levels_bias=scales[0],
                    data_mae=mae4, data_mae_var=mae4, levels_mae=scales[0],
                    data_rmse=rmse4, data_rmse_var=rmse4, levels_rmse=scales[2],
                    title=' CRU - HADcrut - 1920-2020 - ', name_fig='3_1920_2020',
                    s_name=s_name, save=save)


def PlotsMetricsPP(start, end, save = False, lieb = False,
                 noaa20c=True, gpcc=True, cmap=True, prec=True, gpcp=True, ch=True):
    seasons = [1, 4, 7, 10]
    season_name = ['DJF', 'MAM', 'JJA', 'SON']

    scales = [np.linspace(-80, 80, 17),
              np.linspace(0, 80, 9),
              np.linspace(0, 150, 16)]
    cbars = ['RdBu_r', 'BuGn', 'CMRmap_r']


    # n20cr_anom = AnomSeason(t_20cr.sel(time=slice(start + '-01-01', end + '-12-31')))
    # cru_anom = AnomSeason(t_cru.sel(time=slice(start + '-01-01', end + '-12-31')))
    # beic_anom = AnomSeason(t_beic.sel(time=slice(start + '-01-01', end + '-12-31')))
    # ghcn_anom = AnomSeason(t_ghcn.sel(time=slice(start + '-01-01', end + '-12-31')))
    # had_anom = AnomSeason(t_had.sel(time=slice(start + '-01-01', end + '-12-31')))
    # era20_anom = AnomSeason(t_era20.sel(time=slice(start + '-01-01', end + '-12-31')))

    n20cr_clim = ClimSeason(pp_20cr.sel(time=slice(start + '-01-01', end + '-12-31')))
    gpcc_clim = ClimSeason(pp_gpcc.sel(time=slice(start + '-01-01', end + '-12-31')))
    prec_clim = ClimSeason(pp_prec.sel(time=slice(start + '-01-01', end + '-12-31')))
    cmap_clim = ClimSeason(pp_cmap.sel(time=slice(start + '-01-01', end + '-12-31')))
    ch_clim = ClimSeason(pp_ch.sel(time=slice(start + '-01-01', end + '-12-31')))
    gpcp_clim = ClimSeason(pp_gpcp.sel(time=slice(start + '-01-01', end + '-12-31')))

    if lieb:
        lieb_clim = ClimSeason(pp_lieb.sel(time=slice(start + '-01-01', end + '-12-31')))

    for s in seasons:

        if s == 1:
            s_name = 0
        elif s == 4:
            s_name = 1
        elif s == 7:
            s_name = 2
        elif s == 10:
            s_name = 3

        if noaa20c:
            s_20cr = n20cr_clim.groupby('time.month')[s]

        if gpcc:
            s_gpcc = gpcc_clim.groupby('time.month')[s]

        if cmap:
            s_cmap = cmap_clim.groupby('time.month')[s]


        if prec:
            s_prec = prec_clim.groupby('time.month')[s]

        if ch:
            s_ch = ch_clim.groupby('time.month')[s]

        if gpcp:
            s_gpcp = gpcp_clim.groupby('time.month')[s]

        if lieb:
            s_lieb = lieb_clim.groupby('time.month')[s]
            s_lieb['time'] = s_20cr.time.values


        if noaa20c & cmap & gpcc & ch & prec & gpcp:

            if lieb:
                bias1, mae1, rmse1 = MetricasClim(s_lieb, s_20cr)
                bias2, mae2, rmse2 = MetricasClim(s_lieb, s_prec)
                bias3, mae3, rmse3 = MetricasClim(s_lieb, s_ch)
                bias4, mae4, rmse4 = MetricasClim(s_lieb, s_cmap)
                bias5, mae5, rmse5 = MetricasClim(s_lieb, s_gpcp)
                bias6, mae6, rmse6 = MetricasClim(s_lieb, s_gpcc)

            else:
                bias1, mae1, rmse1 = MetricasClim(s_gpcc, s_20cr)
                bias2, mae2, rmse2 = MetricasClim(s_gpcc, s_prec)
                bias3, mae3, rmse3 = MetricasClim(s_gpcc, s_ch)
                bias4, mae4, rmse4 = MetricasClim(s_gpcc, s_cmap)
                bias5, mae5, rmse5 = MetricasClim(s_gpcc, s_gpcp)


            if lieb:
                ref = 'Liebmann'
            else:
                ref = 'GPCC'

            MultPlot(data_bias=bias1, data_bias_var=bias1, levels_bias=scales[0],
                    data_mae=mae1, data_mae_var=mae1, levels_mae=scales[0],
                    data_rmse=rmse1, data_rmse_var=rmse1, levels_rmse=scales[2],
                    title=ref + ' - NOAA20C - 1985-2010 - ', name_fig='1_1985_2010_ref_' + ref,
                    s_name=s_name, save=save,var ='pp',pp=True)

            MultPlot(data_bias=bias2, data_bias_var=bias2, levels_bias=scales[0],
                    data_mae=mae2, data_mae_var=mae2, levels_mae=scales[0],
                    data_rmse=rmse2, data_rmse_var=rmse2, levels_rmse=scales[2],
                    title= ref + ' - PREC - 1985-2010 - ', name_fig='2_1985_2010_ref_' + ref,
                    s_name=s_name, save=save, var ='pp',pp=True)

            MultPlot(data_bias=bias3, data_bias_var=bias3, levels_bias=scales[0],
                    data_mae=mae3, data_mae_var=mae3, levels_mae=scales[0],
                    data_rmse=rmse3, data_rmse_var=rmse3, levels_rmse=scales[2],
                    title=ref +  ' - CHIRPS - 1985-2010 - ', name_fig='3_1985_2010_ref_' + ref,
                    s_name=s_name, save=save, var ='pp',pp=True)

            MultPlot(data_bias=bias4, data_bias_var=bias4, levels_bias=scales[0],
                    data_mae=mae4, data_mae_var=mae4, levels_mae=scales[0],
                    data_rmse=rmse4, data_rmse_var=rmse4, levels_rmse=scales[2],
                    title=ref +  ' - CMAP - 1985-2010', name_fig='4_1985_2010_ref_' + ref,
                    s_name=s_name, save=save, var ='pp',pp=True)

            MultPlot(data_bias=bias5, data_bias_var=bias5, levels_bias=scales[0],
                    data_mae=mae5, data_mae_var=mae5, levels_mae=scales[0],
                    data_rmse=rmse5, data_rmse_var=rmse5, levels_rmse=scales[2],
                    title=ref + ' - CMAP - 1985-2010', name_fig='5_1985_2010_ref_' + ref,
                    s_name=s_name, save=save, var ='pp',pp=True)
            if lieb:
                MultPlot(data_bias=bias6, data_bias_var=bias6, levels_bias=scales[0],
                         data_mae=mae6, data_mae_var=mae6, levels_mae=scales[0],
                         data_rmse=rmse6, data_rmse_var=rmse6, levels_rmse=scales[2],
                         title=ref + ' - GPCC - 1985-2010', name_fig='5_1985_2010_ref_' + ref,
                         s_name=s_name, save=save, var='pp', pp=True)

        elif noaa20c & gpcc & prec & (ch==False) & (gpcp==False) & (cmap==False):
            bias1, mae1, rmse1 = MetricasClim(s_gpcc, s_20cr)
            bias2, mae2, rmse2 = MetricasClim(s_gpcc, s_prec)


            MultPlot(data_bias=bias1, data_bias_var=bias1, levels_bias=scales[0],
                     data_mae=mae1, data_mae_var=mae1, levels_mae=scales[0],
                     data_rmse=rmse1, data_rmse_var=rmse1, levels_rmse=scales[2],
                     title=' GPCC - NOAA20C - 1948-2012 - ', name_fig='1_1948_2012',
                     s_name=s_name, save=save, var='pp', pp=True)

            MultPlot(data_bias=bias2, data_bias_var=bias2, levels_bias=scales[0],
                     data_mae=mae2, data_mae_var=mae2, levels_mae=scales[0],
                     data_rmse=rmse2, data_rmse_var=rmse2, levels_rmse=scales[2],
                     title=' GPCC - PREC - 1948-2012 - ', name_fig='2_1948_2012',
                     s_name=s_name, save=save, var='pp', pp=True)

        elif cmap & gpcc & ch & gpcp & (noaa20c==False) & (prec==False):

            bias3, mae3, rmse3 = MetricasClim(s_gpcc, s_ch)
            bias4, mae4, rmse4 = MetricasClim(s_gpcc, s_cmap)
            bias5, mae5, rmse5 = MetricasClim(s_gpcc, s_gpcp)


            MultPlot(data_bias=bias3, data_bias_var=bias3, levels_bias=scales[0],
                     data_mae=mae3, data_mae_var=mae3, levels_mae=scales[0],
                     data_rmse=rmse3, data_rmse_var=rmse3, levels_rmse=scales[2],
                     title=' GPCC - CHIRPS - 1981-2019 - ', name_fig='3_1981_2019',
                     s_name=s_name, save=save, var='pp', pp=True)

            MultPlot(data_bias=bias4, data_bias_var=bias4, levels_bias=scales[0],
                     data_mae=mae4, data_mae_var=mae4, levels_mae=scales[0],
                     data_rmse=rmse4, data_rmse_var=rmse4, levels_rmse=scales[2],
                     title=' GPCC - CMAP - 1981-2019', name_fig='4_1981_2019',
                     s_name=s_name, save=save, var='pp', pp=True)

            MultPlot(data_bias=bias5, data_bias_var=bias5, levels_bias=scales[0],
                     data_mae=mae5, data_mae_var=mae5, levels_mae=scales[0],
                     data_rmse=rmse5, data_rmse_var=rmse5, levels_rmse=scales[2],
                     title=' GPCC - CMAP - 1981-2019', name_fig='5_1981_2019',
                     s_name=s_name, save=save, var='pp', pp=True)

        elif gpcc & noaa20c & (cmap==False) & (prec==False) & (gpcp==False) & (ch==False):
            # 1920-2020 - todas -{cmap, prec, gpcp, ch}
            bias1, mae1, rmse1 = MetricasClim(s_gpcc, s_20cr)

            MultPlot(data_bias=bias1, data_bias_var=bias1, levels_bias=scales[0],
                     data_mae=mae1, data_mae_var=mae1, levels_mae=scales[0],
                     data_rmse=rmse1, data_rmse_var=rmse1, levels_rmse=scales[2],
                     title=' GPCC - NOAA20C - 1920-2019 - ', name_fig='1_1920_2019',
                     s_name=s_name, save=save, var='pp', pp=True)

########################################################################################################################
# Precipitation ########################################################################################################
#Liebmann
aux = xr.open_dataset(pwd_datos + 'pp_liebmann.nc')
aux = aux.sel(time=slice('1985-01-01','2010-12-31'))
aux = aux.resample(time='1M', skipna=True).mean()
aux = ChangeLons(aux, 'lon')
pp_lieb = aux.sel(lon=slice(275, 330), lat=slice(-50, 20))
pp_lieb = pp_lieb.__mul__(365/12)
pp_lieb = pp_lieb.drop('count')
pp_lieb = pp_lieb.rename({'precip':'pp'})


#NOAA20CR-V3
aux = xr.open_dataset(pwd_datos + 'pp_20CR-V3.nc')
pp_20cr = aux.sel(lon=slice(275, 330), lat=slice(-50, 15))
pp_20cr = pp_20cr.rename({'prate':'pp'})
pp_20cr = pp_20cr.__mul__(86400*(365/12)) #kg/m2/s -> mm/month
pp_20cr = pp_20cr.drop('time_bnds')

#GPCC2018
aux = xr.open_dataset(pwd_datos + 'pp_gpcc.nc')
#interpolado igual que 20cr, los dos son 1x1 pero con distinta grilla
pp_gpcc = aux.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
pp_gpcc = pp_gpcc.sel(lon=slice(275, 330), lat=slice(-50, 15))
pp_gpcc = pp_gpcc.rename({'precip':'pp'})

# PREC
aux = xr.open_dataset(pwd_datos + 'pp_PREC.nc')
pp_prec = aux.sel(lon=slice(275, 330), lat=slice(15, -50))
pp_prec = pp_prec.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
pp_prec = pp_prec.rename({'precip':'pp'})
pp_prec = pp_prec.__mul__(365/12) #mm/day -> mm/month

#CHIRPS
aux = xr.open_dataset(pwd_datos + 'pp_chirps.nc')
aux = ChangeLons(aux, 'longitude')
aux = aux.rename({'precip':'pp', 'latitude':'lat'})
aux = aux.sel(lon=slice(275, 330), lat=slice(-50, 15))
aux = aux.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
pp_ch = aux
del aux

#CMAP
aux = xr.open_dataset(pwd_datos + 'pp_CMAP.nc')
aux = aux.rename({'precip':'pp'})
aux = aux.sel(lon=slice(275, 330), lat=slice(15, -50))
pp_cmap = aux.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
pp_cmap = pp_cmap.__mul__(365/12) #mm/day -> mm/month

# GPCP2.3
aux = xr.open_dataset(pwd_datos + 'pp_gpcp.nc')
aux = aux.rename({'precip':'pp'})
aux = aux.sel(lon=slice(275, 330), lat=slice(-50, 15))
pp_gpcp = aux.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
pp_gpcp = pp_gpcp.drop('lat_bnds')
pp_gpcp = pp_gpcp.drop('lon_bnds')
pp_gpcp = pp_gpcp.drop('time_bnds')
pp_gpcp = pp_gpcp.__mul__(365/12) #mm/day -> mm/month

# plots
PlotsMetricsPP('1985', '2010', save=True,lieb=True)
PlotsMetricsPP('1981', '2019', save=True, noaa20c=False, prec=False)
PlotsMetricsPP('1948', '2012', save=True, cmap=False, gpcp=False, ch=False)
PlotsMetricsPP('1920', '2019', save=True, cmap=False, gpcp=False, ch=False, prec=False)

# Temperature ##########################################################################################################

# 20CR-v3
aux = xr.open_dataset(pwd_datos + 't_20CR-V3.nc')
t_20cr = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))
t_20cr = t_20cr.rename({'air':'temp'})
t_20cr = t_20cr - 273
t_20cr = t_20cr.drop('time_bnds')

# CRU
aux = xr.open_dataset(pwd_datos + 't_cru.nc')
t_cru = ChangeLons(aux)
# interpolado a 1x1
t_cru = t_cru.interp(lat=t_20cr.lat.values, lon=t_20cr.lon.values)
t_cru = t_cru.sel(lon=slice(270, 330), lat=slice(-60, 20),
                  time=slice('1920-01-01','2020-12-31'))
t_cru = t_cru.rename({'tmp':'temp'})
t_cru = t_cru.drop('stn')


#Berkeley Earth etc
aux = xr.open_dataset(pwd_datos + 't_BEIC.nc')
aux = aux.rename({'longitude': 'lon', 'latitude': 'lat', 'temperature':'temp'})
aux = ChangeLons(aux)

aux = aux.interp(lat=t_20cr.lat.values, lon=t_20cr.lon.values)
aux = aux.sel(lon=slice(270, 330), lat=slice(-60, 20), time=slice(1920,2020.999))
aux['time'] = t_cru.time.values
aux['month_number'] = t_cru.time.values[-12:]
t_beic_clim_months = aux.climatology
t_beic = aux.temp
#reconstruyendo?¿
t_beic = t_beic.groupby('time.month') + t_beic_clim_months.groupby('month_number.month').mean()
t_beic = t_beic.drop('month')
t_beic = xr.Dataset(data_vars={'temp': t_beic})

#GHCN
#anom
aux = xr.open_dataset(pwd_datos + 't_ghcn_cams.nc')
aux = aux.interp(lon=t_20cr.lon.values, lat=t_20cr.lat.values)
t_ghcn = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))
t_ghcn = t_ghcn.rename({'air':'temp'})
t_ghcn = t_ghcn - 273

# aux = xr.open_dataset(pwd_datos + 't_ghcn_cams_ltm.nc')
# aux = aux.interp(lon=t_20cr.lon.values, lat=t_20cr.lat.values)
# t_ghcn_clim = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))
# t_ghcn_clim = t_ghcn_clim.rename({'air':'temp'})
# #reconstruyendo?¿
# t_ghcn = t_ghcn.groupby('time.month') + t_ghcn_clim.groupby('time.month').mean()
# t_ghcn = t_ghcn.drop('month')

#HadCRUT
aux = xr.open_dataset(pwd_datos + 't_hadcrut_anom.nc')
aux = ChangeLons(aux,'longitude')
aux = aux.interp(lon=t_20cr.lon.values, latitude=t_20cr.lat.values)
t_had = aux.sel(lon=slice(270, 330), latitude=slice(-60, 20))
t_had = t_had.rename({'tas_mean':'temp', 'latitude':'lat'})
t_had = t_had.sel(time=slice('1920-01-01', '2020-12-31'))


aux = xr.open_dataset(pwd_datos + 't_hadcrut_mean.nc')
aux = ChangeLons(aux)
aux = aux.interp(lon=t_20cr.lon.values, lat=t_20cr.lat.values)
t_had_clim = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))
t_had_clim = t_had_clim.rename({'tem':'temp'})
t_had_clim['time'] = t_cru.time.values[-12:]
#reconstruyendo?¿
t_had = t_had.groupby('time.month') + t_had_clim.groupby('time.month').mean()
t_had = t_had.drop('realization')
t_had = t_had.drop('month')


# ERA-20C
aux = xr.open_dataset(pwd_datos + 't_era20c.nc')
aux = aux.rename({'t2m':'temp', 'latitude':'lat', 'longitude':'lon'})
aux = aux.interp(lon=t_20cr.lon.values, lat=t_20cr.lat.values)
t_era20 = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))
t_era20 = t_era20 - 273

#Plot

PlotsMetricsTemp('1948', '2010', save=True)
PlotsMetricsTemp('1920', '2010', ghcn=False, save=True)
PlotsMetricsTemp('1920', '2020', ghcn=False, era20=False, save=True)



def OpenDatasets(name, interp=False):

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
        pp_20cr = pp_20cr.rename({'prate': 'pp'})
        pp_20cr = pp_20cr.__mul__(86400 * (365 / 12))  # kg/m2/s -> mm/month
        pp_20cr = pp_20cr.drop('time_bnds')

        return pp_20cr
    elif name == 'pp_gpcc':
        # GPCC2018
        aux = xr.open_dataset(pwd_datos + 'pp_gpcc.nc')
        # interpolado igual que 20cr, los dos son 1x1 pero con distinta grilla
        pp_gpcc = aux.sel(lon=slice(270, 330), lat=slice(-50, 20))
        if interp:
            pp_gpcc = aux.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
        pp_gpcc = pp_gpcc.rename({'precip': 'pp'})

        return pp_gpcc
    elif name == 'pp_PREC':
        # PREC
        aux = xr.open_dataset(pwd_datos + 'pp_PREC.nc')
        pp_prec = aux.sel(lon=slice(270, 330), lat=slice(50, -60))
        if interp:
            pp_prec = pp_prec.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
        pp_prec = pp_prec.rename({'precip': 'pp'})
        pp_prec = pp_prec.__mul__(365 / 12)  # mm/day -> mm/month

        return pp_prec
    elif name == 'pp_chirps':
        # CHIRPS
        aux = xr.open_dataset(pwd_datos + 'pp_chirps.nc')
        aux = ChangeLons(aux, 'longitude')
        aux = aux.rename({'precip': 'pp', 'latitude': 'lat'})
        aux = aux.sel(lon=slice(270, 330), lat=slice(-50, 20))
        if interp:
            aux = aux.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
        pp_ch = aux

        return pp_ch
    elif name == 'pp_CMAP':
        # CMAP
        aux = xr.open_dataset(pwd_datos + 'pp_CMAP.nc')
        aux = aux.rename({'precip': 'pp'})
        aux = aux.sel(lon=slice(270, 330), lat=slice(20, -50))
        if interp:
            pp_cmap = aux.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
        pp_cmap = aux.__mul__(365 / 12)  # mm/day -> mm/month

        return pp_cmap
    elif name == 'pp_gpcp':
        # GPCP2.3
        aux = xr.open_dataset(pwd_datos + 'pp_gpcp.nc')
        aux = aux.rename({'precip': 'pp'})
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
        t_20cr = t_20cr.rename({'air': 'temp'})
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
        t_cru = t_cru.rename({'tmp': 'temp'})
        t_cru = t_cru.drop('stn')

        return t_cru
    elif name == 't_BEIC':
        # Berkeley Earth etc
        aux = xr.open_dataset(pwd_datos + 't_BEIC.nc')
        aux = aux.rename({'longitude': 'lon', 'latitude': 'lat', 'temperature': 'temp'})
        aux = ChangeLons(aux)
        aux = aux.sel(lon=slice(270, 330), lat=slice(-60, 20), time=slice(1920, 2020.999))
        if interp:
            aux = aux.interp(lat=t_20cr.lat.values, lon=t_20cr.lon.values)

        aux['time'] = t_cru.time.values
        aux['month_number'] = t_cru.time.values[-12:]
        t_beic_clim_months = aux.climatology
        t_beic = aux.temp
        # reconstruyendo?¿
        t_beic = t_beic.groupby('time.month') + t_beic_clim_months.groupby('month_number.month').mean()
        t_beic = t_beic.drop('month')
        t_beic = xr.Dataset(data_vars={'temp': t_beic})

        return t_beic

    elif name == 't_ghcn_cams':
        # GHCN

        aux = xr.open_dataset(pwd_datos + 't_ghcn_cams.nc')
        aux = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))
        if interp:
            aux = aux.interp(lon=t_20cr.lon.values, lat=t_20cr.lat.values)
        t_ghcn = aux.rename({'air': 'temp'})
        t_ghcn = t_ghcn - 273

        return t_ghcn

    elif name == 't_hadcrut':
        # HadCRUT
        aux = xr.open_dataset(pwd_datos + 't_hadcrut_anom.nc')
        aux = ChangeLons(aux, 'longitude')
        aux = aux.sel(lon=slice(270, 330), latitude=slice(-60, 20))
        if interp:
            aux = aux.interp(lon=t_20cr.lon.values, latitude=t_20cr.lat.values)
        aux = aux.rename({'tas_mean': 'temp', 'latitude': 'lat'})
        t_had = aux.sel(time=slice('1920-01-01', '2020-12-31'))

        aux = xr.open_dataset(pwd_datos + 't_hadcrut_mean.nc')
        aux = ChangeLons(aux)
        aux = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))
        if interp:
            aux = aux.interp(lon=t_20cr.lon.values, lat=t_20cr.lat.values)
        t_had_clim = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))
        aux = aux.rename({'tem': 'temp'})
        aux['time'] = t_cru.time.values[-12:]
        # reconstruyendo?¿
        t_had = t_had.groupby('time.month') + aux.groupby('time.month').mean()
        t_had = t_had.drop('realization')
        t_had = t_had.drop('month')

        return t_had

    elif name == 't_era20c':

        # ERA-20C
        aux = xr.open_dataset(pwd_datos + 't_era20c.nc')
        aux = aux.rename({'t2m': 'temp', 'latitude': 'lat', 'longitude': 'lon'})
        aux = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))
        if interp:
            aux = aux.interp(lon=t_20cr.lon.values, lat=t_20cr.lat.values)
        t_era20 = aux - 273

        return t_era20

