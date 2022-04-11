"""
Calculo del DMI y Nino3.4 en cada miembro de ensamble y lead del NMME-CFSv2
"""

import pandas as pd
import xarray as xr
import numpy as np
from ENSO_IOD_Funciones import MultipleComposite, SelectNMMEFiles

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

def Index(index, std, ms, l):
    """
    Identifica eventos donde la magintud de "index" supere la de "std"

    :param index: [xr Dataset]
    :param std: [xr Dataset]
    :param ms: [int] mes central de la season. eg. 10 para S(9)O(10)N(11)
    :param l: [int] lead
    :return: [xr Dataarray] con la fecha de inicio del pronostico y el valor pronosticado del indice
    """

    """
    Toma el indice en in_month + 1 en cada lead correspondiente debido al 3rm.    
    
    Ej. lead 5 para JJA.
    de esa manera 
    0 1 2 3 4 5 6 7
    E F M A M J J A
    E----->(M-J-J)      | este promedio es debido al 3-month-running mean 
      F----->(J-J-A)    | en Junio estamos viendo el promedio de MJJ, en Julio de JJA, etc
    De esta manera mirando el pronostico ic en Feb
    se SELECCIONA JJA pronosticado 5 meses antes
    Jun--ic en Jan
    Jul--ic en Feb
    Aug--ic en Mar
    """

    print('Forecast for ' + seasons[ms - 7])
    in_month = str(ms - l) #-1
    if in_month == '0':
        in_month = '12'
    # elif in_month == '-1':
    #     in_month = '11'
    print('Lead: ' + str(l))
    print('3rm correction')
    print('Issued at ' + in_month)

    aux = index.groupby('time.month')[int(in_month)].index.sel(L=l)
    aux2 = xr.where(aux.__abs__() > std[l], aux, np.nan)
    index_times = aux2.time[np.where(~np.isnan(aux2.values))]
    INDEX = aux.sel(time=index_times)
    return INDEX
########################################################################################################################
dir_hc = '/pikachu/datos/luciano.andrian/hindcast/'
dir_rt = '/pikachu/datos/luciano.andrian/real_time/'
out_dir = '/datos/luciano.andrian/ncfiles/NMME_CFSv2/index_r/'

leads = [0,1,2,3,4,5,6,7]
seasons = ['JJA', 'JAS', 'ASO', 'SON']
mmonth_seasons = [7,8,9,10]

v = 'sst'

anios_fill = [np.arange(1982,2012),np.arange(2011,2021)]

count_path=0
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
    if count_path==0:
        dates = pd.to_datetime(time[0:-9], format='%Y/%m/%d')
        anios = np.arange(1982, 2011)
    else:
        dates = pd.to_datetime(time[3:], format='%Y/%m/%d')
        anios = np.arange(2011, 2021)


    count_r=0
    for r in range(1, 25): #loop en los miembros de ensamble

        #abre por r .. "*_r1_*"
        files = SelectNMMEFiles(model_name='NCEP-CFSv2', variable=v,
                                dir=path,
                                by_r=True, r=str(r))
        # ordenando por anios
        files = sorted(files, key=lambda x: x.split()[0])

        data = xr.open_mfdataset(files, decode_times=False) #se pudre con los tiempos.... ***
        data = data.rename({'X': 'lon', 'Y': 'lat', 'M': 'r', 'S': 'time'})
        data['time'] = dates # reemplazando con las fechas ***

        # DMI y N34 ####################################################################################################
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
        std_dmi_L = dmi.std('time') * 0.75  # criterio Krishnamurthy

        n34 = xr.Dataset({'index': (('time', 'L'), n34_f)},
                         coords={'time': iodw.time.values, 'L': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
        std_n34_L = n34.std('time')  # criterio Deser

        ################################################################################################################
        seasons = ['JJA', 'JAS', 'ASO', 'SON']
        mmonth_seasons = [7, 8, 9, 10]
        count_ms = 0
        for ms in mmonth_seasons: #loop en las seasons
            count_l = 0
            for l in [0, 1, 2, 3, 4, 5, 6, 7]: #loop en los leads

                iods_df = pd.DataFrame(columns=['DMI', 'Años', 'Mes', 'Lead'], dtype=float)
                n34s_df = pd.DataFrame(columns=['N34', 'Años', 'Mes', 'Lead'], dtype=float)

                # Eventos
                iods = Index(dmi, std_dmi_L.index, ms, l)
                n34s = Index(n34, std_n34_L.index, ms, l)

                # Debido al 3rm en Index
                # solo ms = l = 7 tienen corrimiento de año
                # JJA con lead 7 (explicado en la funcion Index)
                # no existe para 1982, por lo tanto se mira Dec 1982 para JJA 1983 y etc...
                if (ms == 7 ) and (l == 7):
                    fix_year = 1
                else:
                    fix_year = 0

                for c in range(0, len(iods.values)):
                    iods_df = iods_df.append({'DMI': np.around(iods[c].values, 2).T,
                                              'Años': np.around(iods[c]['time.year'].values).T + fix_year,
                                              'Mes': ms,
                                              'Lead': iods.L.values}, ignore_index=True)

                for c in range(0, len(n34s.values)):
                    n34s_df = n34s_df.append({'N34': np.around(n34s[c].values, 2).T,
                                              'Años': np.around(n34s[c]['time.year'].values).T +fix_year,
                                              'Mes': ms,
                                              'Lead': n34s.L.values}, ignore_index=True)

                # clasificacion de los eventos
                Neutral, DMI_sim_pos, DMI_sim_neg, DMI_un_pos, DMI_un_neg, N34_un_pos, N34_un_neg, DMI_pos, \
                DMI_neg, N34_pos, N34_neg = MultipleComposite(var=iodw, n34=n34s_df,
                                                              dmi=iods_df, season=ms - 1,
                                                              start=1982,
                                                              full_season=False, compute_composite=False)
                # Guardado
                if count_l == 0:
                    print('ds 0')
                    ds_l = xr.Dataset(
                        data_vars=dict(
                            Neutral=(['r', 'ms', 'l', 'years'],
                                     np.array(np.where(np.in1d(anios, Neutral), anios, np.nan), ndmin=4)),
                            DMI_sim_pos=(['r', 'ms', 'l', 'years'],
                                         np.array(np.where(np.in1d(anios, DMI_sim_pos), anios, np.nan), ndmin=4)),
                            DMI_sim_neg=(['r', 'ms', 'l', 'years'],
                                         np.array(np.where(np.in1d(anios, DMI_sim_neg), anios, np.nan), ndmin=4)),
                            DMI_un_pos=(['r', 'ms', 'l', 'years'],
                                        np.array(np.where(np.in1d(anios, DMI_un_pos), anios, np.nan), ndmin=4)),
                            DMI_un_neg=(['r', 'ms', 'l', 'years'],
                                        np.array(np.where(np.in1d(anios, DMI_un_neg), anios, np.nan), ndmin=4)),
                            N34_un_pos=(['r', 'ms', 'l', 'years'],
                                        np.array(np.where(np.in1d(anios, N34_un_pos), anios, np.nan), ndmin=4)),
                            N34_un_neg=(['r', 'ms', 'l', 'years'],
                                        np.array(np.where(np.in1d(anios, N34_un_neg), anios, np.nan), ndmin=4)),
                            DMI_pos=(['r', 'ms', 'l', 'years'],
                                     np.array(np.where(np.in1d(anios, DMI_pos), anios, np.nan), ndmin=4)),
                            DMI_neg=(['r', 'ms', 'l', 'years'],
                                     np.array(np.where(np.in1d(anios, DMI_neg), anios, np.nan), ndmin=4)),
                            N34_pos=(['r', 'ms', 'l', 'years'],
                                     np.array(np.where(np.in1d(anios, N34_pos), anios, np.nan), ndmin=4)),
                            N34_neg=(['r', 'ms', 'l', 'years'],
                                     np.array(np.where(np.in1d(anios, N34_neg), anios, np.nan), ndmin=4)),
                        ))
                    count_l += 1

                else:
                    ds2 = xr.Dataset(
                        data_vars=dict(
                            Neutral=(['r', 'ms', 'l', 'years'],
                                     np.array(np.where(np.in1d(anios, Neutral), anios, np.nan), ndmin=4)),
                            DMI_sim_pos=(['r', 'ms', 'l', 'years'],
                                         np.array(np.where(np.in1d(anios, DMI_sim_pos), anios, np.nan), ndmin=4)),
                            DMI_sim_neg=(['r', 'ms', 'l', 'years'],
                                         np.array(np.where(np.in1d(anios, DMI_sim_neg), anios, np.nan), ndmin=4)),
                            DMI_un_pos=(['r', 'ms', 'l', 'years'],
                                        np.array(np.where(np.in1d(anios, DMI_un_pos), anios, np.nan), ndmin=4)),
                            DMI_un_neg=(['r', 'ms', 'l', 'years'],
                                        np.array(np.where(np.in1d(anios, DMI_un_neg), anios, np.nan), ndmin=4)),
                            N34_un_pos=(['r', 'ms', 'l', 'years'],
                                        np.array(np.where(np.in1d(anios, N34_un_pos), anios, np.nan), ndmin=4)),
                            N34_un_neg=(['r', 'ms', 'l', 'years'],
                                        np.array(np.where(np.in1d(anios, N34_un_neg), anios, np.nan), ndmin=4)),
                            DMI_pos=(['r', 'ms', 'l', 'years'],
                                     np.array(np.where(np.in1d(anios, DMI_pos), anios, np.nan), ndmin=4)),
                            DMI_neg=(['r', 'ms', 'l', 'years'],
                                     np.array(np.where(np.in1d(anios, DMI_neg), anios, np.nan), ndmin=4)),
                            N34_pos=(['r', 'ms', 'l', 'years'],
                                     np.array(np.where(np.in1d(anios, N34_pos), anios, np.nan), ndmin=4)),
                            N34_neg=(['r', 'ms', 'l', 'years'],
                                     np.array(np.where(np.in1d(anios, N34_neg), anios, np.nan), ndmin=4)),
                        ))

                    ds_l = xr.concat([ds_l, ds2], dim='l')

            if count_ms == 0:
                ds_ms = ds_l
                count_ms += 1
            else:
                ds_ms = xr.concat([ds_ms, ds_l], dim='ms')

        if count_r == 0:
            ds_r = ds_ms
            count_r += 1
        else:
            ds_r = xr.concat([ds_r, ds_ms], dim='r')

    if count_path == 0:
        ds_p = ds_r
        count_path += 1
    else:
        ds_p = xr.concat([ds_p, ds_r], dim='years')

ds_p.to_netcdf(out_dir + 'dates_dmi_n34_cfsv2.nc')