"""
Calculo del DMI y Nino3.4 en cada miembro de ensamble y lead del NMME-CFSv2
"""

import pandas as pd
import xarray as xr
import numpy as np
from ENSO_IOD_Funciones import SelectNMMEFiles


# Funciones ############################################################################################################
def PreProc(x):
    """
    Preprocesamiento para las series temporales de los indices
    -Anomalia mensual
    -filtrado de tendencia lineal en todo el período
    -3 running mean

    :param x: [xr Dataset] serie temportal
    :return: [array[:,leads=10]] serie procesada en los 10 leads (0-9)
    """
    x = x.groupby('time.month') - x.groupby('time.month').mean()
    x_trend = np.polyfit(range(0, len(x.sst)), x.sst[:, :, 0], deg=1)
    x_detrend = x.sst[:, :, 0] - \
                (x_trend[0, :] * np.repeat([np.arange(1, len(x.sst) + 1)], 10, axis=0).T + x_trend[1, :])

    x_filtered = np.apply_along_axis(lambda m: np.convolve(m, np.ones((3,)) / 3, mode='same'), axis=0,
                                     arr=x_detrend)

    return x_filtered

def fix_calendar(ds, timevar='time'):
    if ds[timevar].attrs['calendar'] == '360':
        ds[timevar].attrs['calendar'] = '360_day'
    return ds
########################################################################################################################
dir_hc = '/pikachu/datos/luciano.andrian/hindcast/'
dir_rt = '/pikachu/datos/luciano.andrian/real_time/'
out_dir = '/datos/luciano.andrian/ncfiles/NMME_CFSv2/DMI_N34_Leads_r/fix/'

leads = [0, 1, 2, 3, 4, 5, 6, 7]
seasons = ['JJA', 'JAS', 'ASO', 'SON']
mmonth_seasons = [7, 8, 9, 10]

v = 'sst'

anios_fill = [np.arange(1982, 2012), np.arange(2011, 2021)]

for ms in mmonth_seasons:
    print(seasons[ms-7])
    r_count = 0
    for r in range(1, 25):  # loop en los miembros de ensamble
        print(r)
        count_path = 0
        for path in [dir_hc, dir_rt]:
            print(path)
            # Fechas
            time = []
            for num in anios_fill[count_path]:
                for m in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
                    if m < 10:
                        time.append(str(num) + '-0' + str(m) + '-01')
                    else:
                        time.append(str(num) + '-' + str(m) + '-01')
            if count_path == 0:
                dates = pd.to_datetime(time[0:-9], format='%Y/%m/%d')
                anios = np.arange(1982, 2011)
                count_path = 1
            else:
                dates = pd.to_datetime(time[3:], format='%Y/%m/%d')
                anios = np.arange(2011, 2021)

            # abre por r .. "*_r1_*"
            files = SelectNMMEFiles(model_name='NCEP-CFSv2', variable=v,
                                    dir=path,
                                    by_r=True, r=str(r))
            # ordenando por anios
            files = sorted(files, key=lambda x: x.split()[0])

            data = xr.open_mfdataset(files, decode_times=False)  # se pudre con los tiempos.... ***
            data = data.rename({'X': 'lon', 'Y': 'lat', 'M': 'r', 'S': 'time'})
            data = xr.decode_cf(fix_calendar(data)) #***

            # DMI y N34 ################################################################################################
            iodw = data.sel(lon=slice(50, 70), lat=slice(-10, 10)).mean(['lon', 'lat'])
            iode = data.sel(lon=slice(90, 110), lat=slice(-10, 0)).mean(['lon', 'lat'])
            n34 = data.sel(lat=slice(-4, 4), lon=slice(190, 240)).mean(['lon', 'lat'])

            # ---------------------- control NANs (a ojo)  ------------------------ #
            # print('#############' + str(r) + '###############' )
            # for L in [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5]:
            #     for t in range(0,351):
            #         if np.isnan(n34.load().sel(L=L).sst[t].values):
            #             print(L)
            #             print(n34.time[t].values)

            # anom mont. detr. 3rm...
            iodw_f = PreProc(iodw)
            iode_f = PreProc(iode)
            n34_f = PreProc(n34)

            dmi = iodw_f - iode_f
            dmi = xr.Dataset({'index': (('time', 'L'), dmi)},
                             coords={'time': iodw.time.values, 'L': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
            # std_dmi_L = dmi.std('time') * 0.75  # criterio Krishnamurthy

            n34 = xr.Dataset({'index': (('time', 'L'), n34_f)},
                             coords={'time': iodw.time.values, 'L': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
            # std_n34_L = n34.std('time')  # criterio Deser

            l_count = 0
            for l in leads:

                ic_month = ms - l  # tomar el mes del centro del trimestre debido al 3rm
                if ic_month == 0:  # JJA y Lead 7
                    ic_month = 12

                dmi_season = dmi.sel(time=dmi.time.dt.month.isin(ic_month), L=l)
                n34_season = n34.sel(time=n34.time.dt.month.isin(ic_month), L=l)

                if l_count == 0:
                    dmi_season_f = dmi_season
                    n34_season_f = n34_season
                    l_count = 1
                else:
                    dmi_season_f = xr.concat([dmi_season_f, dmi_season], dim='time')
                    n34_season_f = xr.concat([n34_season_f, n34_season], dim='time')

            dmi_season_f = dmi_season_f.expand_dims({'r': [r]})
            n34_season_f = n34_season_f.expand_dims({'r': [r]})

            if r_count == 0:
                dmi_season_r_f = dmi_season_f
                n34_season_r_f = n34_season_f
                r_count = 1
            else:
                dmi_season_r_f = xr.merge([dmi_season_r_f, dmi_season_f])
                n34_season_r_f = xr.merge([n34_season_r_f, n34_season_f])

    dmi_season_r_f.to_netcdf(out_dir + seasons[ms - 7] + '_DMI_Leads_r_CFSv2.nc')
    n34_season_r_f.to_netcdf(out_dir + seasons[ms - 7] + '_N34_Leads_r_CFSv2.nc')
    print('Save' + seasons[ms - 7])