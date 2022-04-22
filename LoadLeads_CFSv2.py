"""
Selecciona los conjuntos de leads de NMME CFSv2 en JJA, JAS, ASO y SON en hindcast y realtime
"""
import numpy as np
import xarray as xr
import pandas as pd
from ENSO_IOD_Funciones import SelectNMMEFiles
from datetime import datetime
########################################################################################################################
dir_hc = '/pikachu/datos/osman/nmme/monthly/hindcast/'
dir_rt = '/pikachu/datos/osman/nmme/monthly/real_time/'
out_dir = '/datos/luciano.andrian/ncfiles/NMME_CFSv2/'

# Functions ############################################################################################################
def open_mfdataset_merge_only(paths):
    """
    Los .nc de real_time con xr.open_mfdataset: Resulting object does not have monotonic global indexes along dimension S
    esto es lo mismo pero con mas ram...
    """
    return xr.merge([xr.open_dataset(path,decode_times=False).sel(X=slice(275, 330), Y=slice(-60, 15)) for path in paths])

def fix_calendar(ds, timevar='time'):
    if ds[timevar].attrs['calendar'] == '360':
        ds[timevar].attrs['calendar'] = '360_day'
    return ds
########################################################################################################################

variables = ['tref', 'prec']
anios = np.arange(1982, 2021)

leads = [0,1,2,3,4,5,6,7]
seasons = ['JJA', 'JAS', 'ASO', 'SON']
mmonth_seasons = [7,8,9,10]

"""
Los realtime estan operativos? si es asi el total de eventos va cambiando...
actualmente estan actualizados.
"""

current_year = datetime.now().year
current_month= datetime.now().month
anios_fill = [np.arange(1982, 2011), np.arange(2011, current_year+1)]

for v in variables:
    print(v)
    for ms in mmonth_seasons:
        print(seasons[ms-7])
        r_count=0
        for r in range(1, 25):  # loop en los miembros de ensamble
            print(r)
            count_path=0
            for path in [dir_hc, dir_rt]:
                print(path)
                time = []
                for num in anios_fill[count_path]:
                    for m in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
                        if m < 10:
                            time.append(str(num) + '-0' + str(m) + '-01')
                        else:
                            time.append(str(num) + '-' + str(m) + '-01')
                if count_path == 0:
                    dates = pd.to_datetime(time, format='%Y/%m/%d')
                    anios = np.arange(1982, 2011)
                    count_path = 1
                else:
                    dates = pd.to_datetime(time[:-(12-current_month)], format='%Y/%m/%d')
                    anios = np.arange(2011, 2021)

                # abre por r .. "*_r1_*"
                files = SelectNMMEFiles(model_name='NCEP-CFSv2', variable=v,
                                        dir=path,
                                        by_r=True, r=str(r))

                # algunos archivos estan con nombres mal pero tienen su duplicado con nombre correcto
                # puede que no afecte en xr.open_mfopendataset.
                # filtrando los archivos por cantidad de caracteres
                if path == dir_hc:
                    if r>9:
                        max_len=91
                    else:
                        max_len=90

                    files = [s for s in files if len(s) == max_len]
                else:
                    if r > 9:
                        max_len = 92
                    else:
                        max_len = 91

                    files = [s for s in files if len(s) == max_len]

                # ordenando por anios
                files = sorted(files, key=lambda x: x.split()[0])

                if path==dir_hc:
                    data = xr.open_mfdataset(files, decode_times=False)  # se pudre con los tiempos.... ***
                    data = data.sel(X=slice(275, 330), Y=slice(-60, 15))
                elif path==dir_rt:
                    data = open_mfdataset_merge_only(files) #aca se pudre mas todavia...

                data = data.rename({'X': 'lon', 'Y': 'lat', 'M': 'r', 'S': 'time'})

                data = xr.decode_cf(fix_calendar(data))


                #data['time'] = dates  # reemplazando con las fechas ***

                if path == dir_rt:
                    data = data.sel(time=slice('2011-01-01','2020-12-01'))

                data['L'] = [0,1,2,3,4,5,6,7,8,9]
                data = data.rolling(time=3, center=True).mean()
                """
                3rm, facilita la seleccion y lo hace de manera igual a DMI y N34
                """

                l_count=0
                print('Forecast for ' + seasons[ms - 7])
                for l in leads:
                    ic_month = ms - l  #3rm, se toma el mes del centro del trimestre
                    if ic_month == 0:
                        ic_month = 12
                    elif ic_month == -1:
                        ic_month = 11
                    print('issued at ' + str(ic_month))

                    data_season = data.sel(time=data.time.dt.month.isin(ic_month), L=l)

                    if l_count == 0:
                        data_season_f = data_season
                        l_count = 1
                    else:
                        data_season_f = xr.concat([data_season_f, data_season], dim='time')

                if r_count == 0:
                    data_season_r_f = data_season_f
                    r_count = 1
                else:
                    data_season_r_f = xr.merge([data_season_r_f, data_season_f])

        data_season_r_f.to_netcdf(out_dir + seasons[ms - 7] + '_' + v +'_Leads_r_CFSv2.nc')
        print('Save' + seasons[ms - 7])