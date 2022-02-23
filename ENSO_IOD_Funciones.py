"""
Funciones generales para ENSO vs IOD composite y regression
"""
from itertools import groupby
import xarray as xr
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import statsmodels.formula.api as sm
import os
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

w_dir = '/home/luciano.andrian/doc/salidas/'
out_dir = '/home/luciano.andrian/doc/salidas/ENSO_IOD/'

def MovingBasePeriodAnomaly(data, start=1920, end=2020):
    import xarray as xr
    # first five years
    start_num = start
    start = str(start)

    initial = data.sel(time=slice(start + '-01-01', str(start_num + 5) + '-12-31')).groupby('time.month') - \
              data.sel(time=slice(str(start_num - 14) + '-01-01', str(start_num + 5 + 10) + '-12-31')).groupby(
                  'time.month').mean('time')


    start_num = start_num + 6
    result = initial

    while (start_num != end-4) & (start_num < end-4):

        aux = data.sel(time=slice(str(start_num) + '-01-01', str(start_num + 4) + '-12-31')).groupby('time.month') - \
              data.sel(time=slice(str(start_num - 15) + '-01-01', str(start_num + 4 + 10) + '-12-31')).groupby(
                  'time.month').mean('time')

        start_num = start_num + 5

        result = xr.concat([result, aux], dim='time')

    if start_num > end - 4:
        start_num = start_num - 5

    aux = data.sel(time=slice(str(start_num) + '-01-01', str(start_num + 4) + '-12-31')).groupby('time.month') - \
          data.sel(time=slice(str(end-29) + '-01-01', str(end) + '-12-31')).groupby('time.month').mean('time')

    result = xr.concat([result, aux], dim='time')

    return (result)

def Nino34CPC(data, start=1920, end=2020):

    # Calculates the Niño3.4 index using the CPC criteria.
    # Use ERSSTv5 to obtain exactly the same values as those reported.

    #from Funciones import MovingBasePeriodAnomaly

    start_year = str(start-14)
    end_year = str(end)
    sst = data
    # N34
    ninio34 = sst.sel(lat=slice(4.0, -4.0), lon=slice(190, 240), time=slice(start_year+'-01-01', end_year + '-12-31'))
    ninio34 = ninio34.sst.mean(['lon', 'lat'], skipna=True)

    # compute monthly anomalies
    ninio34 = MovingBasePeriodAnomaly(data=ninio34, start=start, end=end)

    # compute 5-month running mean
    ninio34_filtered = np.convolve(ninio34, np.ones((3,)) / 3, mode='same')  #
    ninio34_f = xr.DataArray(ninio34_filtered, coords=[ninio34.time.values], dims=['time'])

    aux = abs(ninio34_f) > 0.5
    results = []
    for k, g in groupby(enumerate(aux.values), key=lambda x: x[1]):
        if k:
            g = list(g)
            results.append([g[0][0], len(g)])

    n34 = []
    n34_df = pd.DataFrame(columns=['N34', 'Años', 'Mes'], dtype=float)
    for m in range(0, len(results)):
        # True values
        len_true = results[m][1]

        # True values for at least 5 consecutive seasons
        if len_true >= 5:
            a = results[m][0]
            n34.append([np.arange(a, a + results[m][1]), ninio34_f[np.arange(a, a + results[m][1])].values])

            for l in range(0, len_true):
                if l < (len_true - 2):
                    main_month_num = results[m][0] + 1 + l
                    if main_month_num != 1210:
                        n34_df = n34_df.append({'N34': np.around(ninio34_f[main_month_num].values, 2),
                                            'Años': np.around(ninio34_f[main_month_num]['time.year'].values),
                                            'Mes': np.around(ninio34_f[main_month_num]['time.month'].values)},
                                           ignore_index=True)

    return ninio34_f, n34, n34_df

def WaveFilter(serie, harmonic):

    import numpy as np

    sum = 0
    sam = 0
    N = np.size(serie)

    sum = 0
    sam = 0

    for j in range(N):
        sum = sum + serie[j] * np.sin(harmonic * 2 * np.pi * j / N)
        sam = sam + serie[j] * np.cos(harmonic * 2 * np.pi * j / N)

    A = 2*sum/N
    B = 2*sam/N

    xs = np.zeros(N)

    for j in range(N):
        xs[j] = A * np.sin(2 * np.pi * harmonic * j / N) + B * np.cos(2 * np.pi * harmonic * j / N)

    fil = serie - xs
    return(fil)

def DMIndex(iodw, iode, sst_anom_sd=True, xsd=0.5):

    import numpy as np
    from itertools import groupby
    import pandas as pd

    limitsize = len(iodw) - 2

    # dipole mode index
    dmi = iodw - iode

    # criteria
    western_sign = np.sign(iodw)
    eastern_sign = np.sign(iode)
    opposite_signs = western_sign != eastern_sign

    sd = np.std(dmi) * xsd
    print(str(sd))
    sdw = np.std(iodw.values) * xsd
    sde = np.std(iode.values) * xsd

    results = []
    for k, g in groupby(enumerate(opposite_signs.values), key=lambda x: x[1]):
        if k:
            g = list(g)
            results.append([g[0][0], len(g)])

    iods = pd.DataFrame(columns=['DMI', 'Años', 'Mes'], dtype=float)
    dmi_raw = []
    for m in range(0, len(results)):
        # True values
        len_true = results[m][1]

        # True values for at least 3 consecutive seasons
        if len_true >= 3:

            for l in range(0, len_true):

                if l < (len_true - 2):

                    main_month_num = results[m][0] + 1 + l
                    if main_month_num != limitsize:
                        main_month_name = dmi[main_month_num]['time.month'].values  # "name" 1 2 3 4 5

                        main_season = dmi[main_month_num]
                        b_season = dmi[main_month_num - 1]
                        a_season = dmi[main_month_num + 1]

                        # abs(dmi) > sd....(0.5*sd)
                        aux = (abs(main_season.values) > sd) & \
                              (abs(b_season) > sd) & \
                              (abs(a_season) > sd)

                        if sst_anom_sd:
                            if aux:
                                sstw_main = iodw[main_month_num]
                                sstw_b = iodw[main_month_num - 1]
                                sstw_a = iodw[main_month_num + 1]
                                #
                                aux2 = (abs(sstw_main) > sdw) & \
                                       (abs(sstw_b) > sdw) & \
                                       (abs(sstw_a) > sdw)
                                #
                                sste_main = iode[main_month_num]
                                sste_b = iode[main_month_num - 1]
                                sste_a = iode[main_month_num + 1]

                                aux3 = (abs(sste_main) > sde) & \
                                       (abs(sste_b) > sde) & \
                                       (abs(sste_a) > sde)

                                if aux3 & aux2:
                                    iods = iods.append({'DMI': np.around(dmi[main_month_num].values, 2),
                                                        'Años': np.around(dmi[main_month_num]['time.year'].values),
                                                        'Mes': np.around(dmi[main_month_num]['time.month'].values)},
                                                       ignore_index=True)

                                    a = results[m][0]
                                    dmi_raw.append([np.arange(a, a + results[m][1]),
                                                    dmi[np.arange(a, a + results[m][1])].values])


                        else:
                            if aux:
                                iods = iods.append({'DMI': np.around(dmi[main_month_num].values, 2),
                                                    'Años': np.around(dmi[main_month_num]['time.year'].values),
                                                    'Mes': np.around(dmi[main_month_num]['time.month'].values)},
                                                   ignore_index=True)

    return iods, dmi_raw

def DMI(per = 0, filter_bwa = True, filter_harmonic = True,
        filter_all_harmonic=True, harmonics = [],
        start_per=1920, end_per=2020):


    western_io = slice(50, 70) # definicion tradicional

    start_per = str(start_per)
    end_per = str(end_per)

    if per == 2:
        movinganomaly = True
        start_year = '1906'
        end_year = '2020'
        change_baseline = False
        start_year2 = '1920'
        end_year2 = '2020_30r5'
        print('30r5')
    else:
        movinganomaly = False
        start_year = start_per
        end_year = end_per
        change_baseline = False
        start_year2 = '1920'
        end_year2 = end_per
        print('All')

    ##################################### DATA #####################################
    # ERSSTv5
    sst = xr.open_dataset("/pikachu/datos4/Obs/sst/sst.mnmean_2020.nc")
    dataname = 'ERSST'
    ##################################### Pre-processing #####################################
    iodw = sst.sel(lat=slice(10.0, -10.0), lon=western_io,
                       time=slice(start_year + '-01-01', end_year + '-12-31'))
    iodw = iodw.sst.mean(['lon', 'lat'], skipna=True)
    iodw2 = iodw
    if per == 2:
        iodw2 = iodw2[168:]
    # -----------------------------------------------------------------------------------#
    iode = sst.sel(lat=slice(0, -10.0), lon=slice(90, 110),
                   time=slice(start_year + '-01-01', end_year + '-12-31'))
    iode = iode.sst.mean(['lon', 'lat'], skipna=True)
    # -----------------------------------------------------------------------------------#
    bwa = sst.sel(lat=slice(20.0, -20.0), lon=slice(40, 110),
                  time=slice(start_year + '-01-01', end_year + '-12-31'))
    bwa = bwa.sst.mean(['lon', 'lat'], skipna=True)
    # ----------------------------------------------------------------------------------#

    if movinganomaly:
        iodw = MovingBasePeriodAnomaly(iodw)
        iode = MovingBasePeriodAnomaly(iode)
        bwa = MovingBasePeriodAnomaly(bwa)
    else:
        # change baseline
        if change_baseline:
            iodw = iodw.groupby('time.month') - \
                   iodw.sel(time=slice(start_year2 + '-01-01', end_year2 + '-12-31')).groupby('time.month').mean(
                       'time')

            iode = iode.groupby('time.month') - \
                   iode.sel(time=slice(start_year2 + '-01-01', end_year2 + '-12-31')).groupby('time.month').mean(
                       'time')

            bwa = bwa.groupby('time.month') - \
                  bwa.sel(time=slice(start_year2 + '-01-01', end_year2 + '-12-31')).groupby('time.month').mean(
                      'time')
            print('baseline: ' + str(start_year2) + ' - ' + str(end_year2))
        else:
            print('baseline: All period')
            iodw = iodw.groupby('time.month') - iodw.groupby('time.month').mean('time', skipna=True)
            iode = iode.groupby('time.month') - iode.groupby('time.month').mean('time', skipna=True)
            bwa = bwa.groupby('time.month') - bwa.groupby('time.month').mean('time', skipna=True)

    # ----------------------------------------------------------------------------------#
    # Detrend
    iodw_trend = np.polyfit(range(0, len(iodw)), iodw, deg=1)
    iodw = iodw - (iodw_trend[0] * range(0, len(iodw)) + iodw_trend[1])
    # ----------------------------------------------------------------------------------#
    iode_trend = np.polyfit(range(0, len(iode)), iode, deg=1)
    iode = iode - (iode_trend[0] * range(0, len(iode)) + iode_trend[1])
    # ----------------------------------------------------------------------------------#
    bwa_trend = np.polyfit(range(0, len(bwa)), bwa, deg=1)
    bwa = bwa - (bwa_trend[0] * range(0, len(bwa)) + bwa_trend[1])
    # ----------------------------------------------------------------------------------#

    # 3-Month running mean
    iodw_filtered = np.convolve(iodw, np.ones((3,)) / 3, mode='same')
    iode_filtered = np.convolve(iode, np.ones((3,)) / 3, mode='same')
    bwa_filtered = np.convolve(bwa, np.ones((3,)) / 3, mode='same')

    # Common preprocessing, for DMIs other than SY2003a
    iode_3rm = iode_filtered
    iodw_3rm = iodw_filtered

    #################################### follow SY2003a #######################################

    # power spectrum
    # aux = FFT2(iodw_3rm, maxVar=20, maxA=15).sort_values('Variance', ascending=False)
    # aux2 = FFT2(iode_3rm, maxVar=20, maxA=15).sort_values('Variance', ascending=False)

    # filtering harmonic
    if filter_harmonic:
        if filter_all_harmonic:
            for harmonic in range(15):
                iodw_filtered = WaveFilter(iodw_filtered, harmonic)
                iode_filtered = WaveFilter(iode_filtered, harmonic)
            else:
                for harmonic in harmonics:
                    iodw_filtered = WaveFilter(iodw_filtered, harmonic)
                    iode_filtered = WaveFilter(iode_filtered, harmonic)

    ## max corr. lag +3 in IODW
    ## max corr. lag +6 in IODE

    # ----------------------------------------------------------------------------------#
    # ENSO influence
    # pre processing same as before
    if filter_bwa:
        ninio3 = sst.sel(lat=slice(5.0, -5.0), lon=slice(210, 270),
                         time=slice(start_year + '-01-01', end_year + '-12-31'))
        ninio3 = ninio3.sst.mean(['lon', 'lat'], skipna=True)

        if movinganomaly:
            ninio3 = MovingBasePeriodAnomaly(ninio3)
        else:
            if change_baseline:
                ninio3 = ninio3.groupby('time.month') - \
                         ninio3.sel(time=slice(start_year2 + '-01-01', end_year2 + '-12-31')).groupby(
                             'time.month').mean(
                             'time')

            else:

                ninio3 = ninio3.groupby('time.month') - ninio3.groupby('time.month').mean('time', skipna=True)

            trend = np.polyfit(range(0, len(ninio3)), ninio3, deg=1)
            ninio3 = ninio3 - (trend[0] * range(0, len(ninio3)) +trend[1])

        # 3-month running mean
        ninio3_filtered = np.convolve(ninio3, np.ones((3,)) / 3, mode='same')

        # ----------------------------------------------------------------------------------#
        # removing BWA effect
        # lag de maxima corr coincide para las dos bases de datos.
        lag = 3
        x = pd.DataFrame({'bwa': bwa_filtered[lag:], 'ninio3': ninio3_filtered[:-lag]})
        result = sm.ols(formula='bwa~ninio3', data=x).fit()
        recta = result.params[1] * ninio3_filtered + result.params[0]
        iodw_f = iodw_filtered - recta

        lag = 6
        x = pd.DataFrame({'bwa': bwa_filtered[lag:], 'ninio3': ninio3_filtered[:-lag]})
        result = sm.ols(formula='bwa~ninio3', data=x).fit()
        recta = result.params[1] * ninio3_filtered + result.params[0]
        iode_f = iode_filtered - recta
        print('BWA filtrado')
    else:
        iodw_f = iodw_filtered
        iode_f = iode_filtered
        print('BWA no filtrado')
    # ----------------------------------------------------------------------------------#

    # END processing
    if movinganomaly:
        iodw_3rm = xr.DataArray(iodw_3rm, coords=[iodw.time.values], dims=['time'])
        iode_3rm = xr.DataArray(iode_3rm, coords=[iodw.time.values], dims=['time'])

        iodw_f = xr.DataArray(iodw_f, coords=[iodw.time.values], dims=['time'])
        iode_f = xr.DataArray(iode_f, coords=[iodw.time.values], dims=['time'])
        start_year = '1920'
    else:
        iodw_3rm = xr.DataArray(iodw_3rm, coords=[iodw2.time.values], dims=['time'])
        iode_3rm = xr.DataArray(iode_3rm, coords=[iodw2.time.values], dims=['time'])

        iodw_f = xr.DataArray(iodw_f, coords=[iodw2.time.values], dims=['time'])
        iode_f = xr.DataArray(iode_f, coords=[iodw2.time.values], dims=['time'])

    ####################################### compute DMI #######################################

    dmi_sy_full, dmi_raw = DMIndex(iodw_f, iode_f)

    return dmi_sy_full, dmi_raw, (iodw_f-iode_f)#, iodw_f - iode_f, iodw_f, iode_f

def PlotEnso_Iod(dmi, ninio, title, fig_name = 'fig_enso_iod', out_dir=out_dir, save=False):
    from numpy import ma
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    im = plt.scatter(x=dmi, y=ninio, marker='o', s=20, edgecolor='black', color='gray')

    plt.ylim((-4, 4));
    plt.xlim((-4, 4))
    plt.axhspan(-0.31, 0.31, alpha=0.2, color='black', zorder=0)
    plt.axvspan(-0.5, 0.5, alpha=0.2, color='black', zorder=0)
    # ax.grid(True)
    fig.set_size_inches(6, 6)
    plt.xlabel('IOD')
    plt.ylabel('Niño 3.4')

    plt.text(-3.8, 3.4, '    EN/IOD-', dict(size=10))
    plt.text(-.1, 3.4, 'EN', dict(size=10))
    plt.text(+2.6, 3.4, ' EN/IOD+', dict(size=10))
    plt.text(+2.6, -.1, 'IOD+', dict(size=10))
    plt.text(+2.3, -3.4, '    LN/IOD+', dict(size=10))
    plt.text(-.1, -3.4, 'LN', dict(size=10))
    plt.text(-3.8, -3.4, ' LN/IOD-', dict(size=10))
    plt.text(-3.8, -.1, 'IOD-', dict(size=10))
    plt.title(title)
    if save:
        plt.savefig(out_dir + 'ENSO_IOD'+ fig_name + '.jpg')
    else:
        plt.show()


def SelectYears(df, name_var, main_month=1, full_season=False):

    if full_season:
        print('Full Season JJASON')
        aux = pd.DataFrame({'Ind': df.where(df.Mes.isin([7, 8, 9, 10, 11]))[name_var],
                            'Años': df.where(df.Mes.isin([7, 8, 9, 10, 11]))['Años'],
                            'Mes': df.where(df.Mes.isin([7, 8, 9, 10, 11]))['Mes']})
        mmin, mmax = 6, 11

    else:
        aux = pd.DataFrame({'Ind': df.where(df.Mes.isin([main_month]))[name_var],
                            'Años': df.where(df.Mes.isin([main_month]))['Años'],
                            'Mes': df.where(df.Mes.isin([main_month]))['Mes']})
        mmin, mmax = main_month - 1, main_month + 1

        if main_month == 1:
            mmin, mmax = 12, 2
        elif main_month == 12:
            mmin, mmax = 11, 1

    return aux.dropna(), mmin, mmax

def ClassifierEvents(df, full_season=False):
    if full_season:
        print('full season')
        df_pos = set(df.Años.values[np.where(df['Ind'] > 0)])
        df_neg = set(df.Años.values[np.where(df['Ind'] < 0)])
    else:
        df_pos = df.Años.values[np.where(df['Ind'] > 0)]
        df_neg = df.Años.values[np.where(df['Ind'] < 0)]

    return df_pos, df_neg

def is_months(month, mmin, mmax):
    return (month >= mmin) & (month <= mmax)

def NeutralEvents(df, mmin, mmax, start=1920, end = 2020, double=False, df2=None, var_original=None):

    x = np.arange(start, end + 1, 1)

    start = str(start)
    end = str(end)

    mask = np.in1d(x, df.Años.values, invert=True)
    if mmax ==1: #NDJ
        print("NDJ Special")
        neutro = var_original.sel(time=var_original.time.dt.year.isin(x[mask]))
        neutro_1 = var_original.sel(time=var_original.time.dt.year.isin(x[mask]+1))
        if double:
            mask = np.in1d(x, df2.Años.values, invert=True)
            neutro = neutro.sel(time=neutro.time.dt.year.isin(x[mask]))
            neutro_1 = neutro_1.sel(time=neutro_1.time.dt.year.isin(x[mask]))

        neutro = neutro.sel(time=is_months(month=neutro['time.month'], mmin=11, mmax=12))
        neutro_1 = neutro_1.sel(time=neutro_1.time.dt.month.isin(1))
        neutro = xr.merge([neutro, neutro_1])
        neutro = neutro.mean(['time'], skipna=True)

    elif mmin == 12: #DJF
        print("DJF Special")
        neutro = var_original.sel(time=var_original.time.dt.year.isin(x[mask]))
        neutro_1 = var_original.sel(time=var_original.time.dt.year.isin(x[mask]-1))
        if double:
            mask = np.in1d(x, df2.Años.values, invert=True)
            neutro = neutro.sel(time=neutro.time.dt.year.isin(x[mask]))
            neutro_1 = neutro_1.sel(time=neutro_1.time.dt.year.isin(x[mask]))

        neutro = neutro.sel(time=is_months(month=neutro['time.month'], mmin=1, mmax=2))
        neutro_1 = neutro_1.sel(time=neutro_1.time.dt.month.isin(12))
        neutro = xr.merge([neutro, neutro_1])
        neutro = neutro.mean(['time'], skipna=True)

    else:
        mask = np.in1d(x, df.Años.values, invert=True)
        neutro = var_original.sel(time=var_original.time.dt.year.isin(x[mask]))
        if double:
            mask = np.in1d(x, df2.Años.values, invert=True)
            neutro = neutro.sel(time=neutro.time.dt.year.isin(x[mask]))
            neutro_years = list(set(neutro.time.dt.year.values))
        neutro = neutro.sel(time=is_months(month=neutro['time.month'], mmin=mmin, mmax=mmax))
        neutro = neutro.mean(['time'], skipna=True)

    return neutro, neutro_years

def Composite(original_data, index_pos, index_neg, mmin, mmax):
    comp_field_pos=0
    comp_field_neg=0

    if len(index_pos) != 0:
        if mmax == 1:
            print('NDJ Special')
            comp_field_pos = original_data.sel(time=original_data.time.dt.year.isin(index_pos))
            comp_field_pos_1 = original_data.sel(time=original_data.time.dt.year.isin(index_pos+1))

            comp_field_pos = comp_field_pos.sel(
                time=is_months(month=comp_field_pos['time.month'], mmin=11, mmax=12))
            comp_field_pos_1 = comp_field_pos_1.sel(time=comp_field_pos_1.time.dt.month.isin(1))

            comp_field_pos = xr.merge([comp_field_pos, comp_field_pos_1])
            if len(comp_field_pos.time) != 0:
                comp_field_pos = comp_field_pos.mean(['time'], skipna=True)
            else:
                comp_field_pos = comp_field_pos.drop_dims(['time'])

        elif mmin == 12:
            print('DJF Special')
            comp_field_pos = original_data.sel(time=original_data.time.dt.year.isin(index_pos))
            comp_field_pos_1 = original_data.sel(time=original_data.time.dt.year.isin(index_pos - 1))

            comp_field_pos = comp_field_pos.sel(
                time=is_months(month=comp_field_pos['time.month'], mmin=1, mmax=2))
            comp_field_pos_1 = comp_field_pos_1.sel(time=comp_field_pos_1.time.dt.month.isin(2))

            comp_field_pos = xr.merge([comp_field_pos, comp_field_pos_1])
            if len(comp_field_pos.time) != 0:
                comp_field_pos = comp_field_pos.mean(['time'], skipna=True)
            else:
                comp_field_pos = comp_field_pos.drop_dims(['time'])

        else:
            comp_field_pos = original_data.sel(time=original_data.time.dt.year.isin([index_pos]))
            comp_field_pos = comp_field_pos.sel(
                time=is_months(month=comp_field_pos['time.month'], mmin=mmin, mmax=mmax))
            if len(comp_field_pos.time) != 0:
                comp_field_pos = comp_field_pos.mean(['time'], skipna=True)
            else:
                comp_field_pos = comp_field_pos.drop_dims(['time'])


    if len(index_neg) != 0:
        if mmax == 1:
            print('NDJ Special')
            comp_field_neg = original_data.sel(time=original_data.time.dt.year.isin(index_neg))
            comp_field_neg_1 = original_data.sel(time=original_data.time.dt.year.isin(index_neg + 1))

            comp_field_neg = comp_field_neg.sel(
                time=is_months(month=comp_field_neg['time.month'], mmin=11, mmax=12))
            comp_field_neg_1 = comp_field_neg_1.sel(time=comp_field_neg_1.time.dt.month.isin(1))

            comp_field_neg = xr.merge([comp_field_neg, comp_field_neg_1])
            if (len(comp_field_neg.time) != 0):
                comp_field_neg = comp_field_neg.mean(['time'], skipna=True)
            else:
                comp_field_neg = comp_field_neg.drop_dmis(['time'])

        elif mmin == 12:
            print('DJF Special')
            comp_field_neg = original_data.sel(time=original_data.time.dt.year.isin(index_neg))
            comp_field_neg_1 = original_data.sel(time=original_data.time.dt.year.isin(index_neg - 1))

            comp_field_neg = comp_field_neg.sel(
                time=is_months(month=comp_field_neg['time.month'], mmin=1, mmax=2))
            comp_field_neg_1 = comp_field_neg_1.sel(time=comp_field_neg_1.time.dt.month.isin(2))

            comp_field_neg = xr.merge([comp_field_neg, comp_field_neg_1])
            if len(comp_field_neg.time) != 0:
                comp_field_neg = comp_field_neg.mean(['time'], skipna=True)
            else:
                comp_field_neg = comp_field_neg.drop_dims(['time'])

        else:
            comp_field_neg = original_data.sel(time=original_data.time.dt.year.isin([index_neg]))
            comp_field_neg = comp_field_neg.sel(time=is_months(month=comp_field_neg['time.month'],
                                                               mmin=mmin, mmax=mmax))
            if len(comp_field_neg.time) != 0:
                comp_field_neg = comp_field_neg.mean(['time'], skipna=True)
            else:
                comp_field_neg = comp_field_neg.drop_dims(['time'])

    return comp_field_pos, comp_field_neg

def c_diff(arr, h, dim, cyclic=False):
    # compute derivate of array variable respect to h associated to dim
    # adapted from kuchaale script
    ndim = arr.ndim
    lst = [i for i in range(ndim)]

    lst[dim], lst[0] = lst[0], lst[dim]
    rank = lst
    arr = np.transpose(arr, tuple(rank))

    if ndim == 3:
        shp = (arr.shape[0] - 2, 1, 1)
    elif ndim == 4:
        shp = (arr.shape[0] - 2, 1, 1, 1)

    d_arr = np.copy(arr)
    if not cyclic:
        d_arr[0, ...] = (arr[1, ...] - arr[0, ...]) / (h[1] - h[0])
        d_arr[-1, ...] = (arr[-1, ...] - arr[-2, ...]) / (h[-1] - h[-2])
        d_arr[1:-1, ...] = (arr[2:, ...] - arr[0:-2, ...]) / np.reshape(h[2:] - h[0:-2], shp)

    elif cyclic:
        d_arr[0, ...] = (arr[1, ...] - arr[-1, ...]) / (h[1] - h[-1])
        d_arr[-1, ...] = (arr[0, ...] - arr[-2, ...]) / (h[0] - h[-2])
        d_arr[1:-1, ...] = (arr[2:, ...] - arr[0:-2, ...]) / np.reshape(h[2:] - h[0:-2], shp)

    d_arr = np.transpose(d_arr, tuple(rank))

    return d_arr

def WAF(psiclm, psiaa, lon, lat,reshape=True, variable='var'):
    #agregar xr=True

    if reshape:
        psiclm=psiclm[variable].values.reshape(1,len(psiclm.lat),len(psiclm.lon))
        psiaa = psiaa[variable].values.reshape(1, len(psiaa.lat), len(psiaa.lon))

    lon=lon.values
    lat=lat.values

    [xxx, nlats, nlons] = psiaa.shape  # get dimensions
    a = 6400000
    coslat = np.cos(lat * np.pi / 180)

    # climatological wind at psi level
    dpsiclmdlon = c_diff(psiclm, lon, 2)
    dpsiclmdlat = c_diff(psiclm, lat, 1)

    uclm = -1 * dpsiclmdlat
    vclm = dpsiclmdlon
    magU = np.sqrt(np.add(np.power(uclm, 2), np.power(vclm, 2)))

    dpsidlon = c_diff(psiaa, lon, 2)
    ddpsidlonlon = c_diff(dpsidlon, lon, 2)
    dpsidlat = c_diff(psiaa, lat, 1)
    ddpsidlatlat = c_diff(dpsidlat, lat, 1)
    ddpsidlatlon = c_diff(dpsidlat, lon, 2)

    termxu = dpsidlon * dpsidlon - psiaa * ddpsidlonlon
    termxv = dpsidlon * dpsidlat - ddpsidlatlon * psiaa
    termyv = dpsidlat * dpsidlat - psiaa * ddpsidlatlat

    # 0.2101 is the scale of p VER!!!
    coeff1 = np.transpose(np.tile(coslat, (nlons, 1))) * (0.2101) / (2 * magU)
    # x-component
    px = coeff1 / (a * a * np.transpose(np.tile(coslat, (nlons, 1)))) * (
            uclm * termxu / np.transpose(np.tile(coslat, (nlons, 1))) + (vclm * termxv))
    # y-component
    py = coeff1 / (a * a) * (uclm / np.transpose(np.tile(coslat, (nlons, 1))) * termxv + (vclm * termyv))

    return px, py

def MultipleComposite(var, n34, dmi, season,start = 1920, full_season=False, compute_composite=False):

    seasons = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
               'JJA','JAS', 'ASO', 'SON', 'OND', 'NDJ']


    def check(x):
        if x is None:
            x = [0]
            return x
        else:
            if len(x) == 0:
                x = [0]
                return x
        return x

    if full_season:
        main_month_name = 'JJASON'
        main_month = None
    else:
        main_month, main_month_name = len(seasons[:season]) + 1, seasons[season]

    print(main_month_name)

    N34, N34_mmin, N34_mmax = SelectYears(df=n34, name_var='N34',
                                          main_month=main_month, full_season=full_season)
    DMI, DMI_mmin, DMI_mmax = SelectYears(df=dmi, name_var='DMI',
                                          main_month=main_month, full_season=full_season)
    DMI_sim_pos = [0,0]
    DMI_sim_neg = [0,0]
    DMI_un_pos = [0,0]
    DMI_un_neg = [0,0]
    DMI_pos = [0,0]
    DMI_neg = [0,0]
    N34_sim_pos = [0,0]
    N34_sim_neg = [0,0]
    N34_un_pos = [0,0]
    N34_un_neg = [0,0]
    N34_pos = [0,0]
    N34_neg = [0,0]
    All_neutral = [0, 0]

    if (len(DMI) != 0) & (len(N34) != 0):
        # All events
        DMI_pos, DMI_neg = ClassifierEvents(DMI, full_season=full_season)
        N34_pos, N34_neg = ClassifierEvents(N34, full_season=full_season)

        # both neutral, DMI and N34
        if compute_composite:
            All_neutral = NeutralEvents(df=DMI, mmin=DMI_mmin, mmax=DMI_mmax, start=start,
                                        df2=N34, double=True, var_original=var)[0]

        else:
            All_neutral = NeutralEvents(df=DMI, mmin=DMI_mmin, mmax=DMI_mmax, start=start,
                                        df2=N34, double=True, var_original=var)[1]


        # Simultaneous events
        sim_events = np.intersect1d(N34.Años.values, DMI.Años.values)

        try:
            # Simultaneos events
            DMI_sim = DMI.where(DMI.Años.isin(sim_events)).dropna()
            #N34_sim = N34.where(N34.Años.isin(sim_events)).dropna()
            DMI_sim_pos, DMI_sim_neg = ClassifierEvents(DMI_sim)
            #N34_sim_pos, N34_sim_neg = ClassifierEvents(N34_sim)

            # Unique events
            DMI_un = DMI.where(-DMI.Años.isin(sim_events)).dropna()
            N34_un = N34.where(-N34.Años.isin(sim_events)).dropna()

            DMI_un_pos, DMI_un_neg = ClassifierEvents(DMI_un)
            N34_un_pos, N34_un_neg = ClassifierEvents(N34_un)

            if compute_composite:
                print('Making composites...')
                # ------------------------------------ SIMULTANEUS ---------------------------------------------#
                DMI_sim = Composite(original_data=var, index_pos=DMI_sim_pos, index_neg=DMI_sim_neg,
                                    mmin=DMI_mmin, mmax=DMI_mmax)

                # ------------------------------------ UNIQUES -------------------------------------------------#
                DMI_un = Composite(original_data=var, index_pos=DMI_un_pos, index_neg=DMI_un_neg,
                                   mmin=DMI_mmin, mmax=DMI_mmax)

                N34_un = Composite(original_data=var, index_pos=N34_un_pos, index_neg=N34_un_neg,
                                   mmin=N34_mmin, mmax=N34_mmax)
            else:
                print('Only dates, no composites')
                DMI_sim = None
                DMI_un = None
                N34_un = None

        except:
            DMI_sim = None
            DMI_un = None
            N34_un = None
            DMI_sim_pos = None
            DMI_sim_neg = None
            DMI_un_pos = None
            DMI_un_neg = None
            print('Only uniques events[3][4]')

        if compute_composite:
            # ------------------------------------ ALL ---------------------------------------------#
            dmi_comp = Composite(original_data=var, index_pos=list(DMI_pos), index_neg=list(DMI_neg),
                                 mmin=DMI_mmin, mmax=DMI_mmax)
            N34_comp = Composite(original_data=var, index_pos=list(N34_pos), index_neg=list(N34_neg),
                                 mmin=N34_mmin, mmax=N34_mmax)
        else:
            dmi_comp=None
            N34_comp=None

    DMI_sim_pos = check(DMI_sim_pos)
    DMI_sim_neg = check(DMI_sim_neg)
    DMI_un_pos = check(DMI_un_pos)
    DMI_un_neg = check(DMI_un_neg)
    DMI_pos = check(DMI_pos)
    DMI_neg = check(DMI_neg)

    N34_sim_pos = check(N34_sim_pos)
    N34_sim_neg = check(N34_sim_neg)
    N34_un_pos = check(N34_un_pos)
    N34_un_neg = check(N34_un_neg)
    N34_pos = check(N34_pos)
    N34_neg = check(N34_neg)

    All_neutral = check(All_neutral)


    if compute_composite:
        print('test')
        return DMI_sim, DMI_un, N34_un, dmi_comp, N34_comp, All_neutral, DMI_sim_pos, DMI_sim_neg, \
               DMI_un_pos, DMI_un_neg, N34_un_pos, N34_un_neg, DMI_pos, DMI_neg, N34_pos, N34_neg
    else:
        return list(All_neutral),\
               list(set(DMI_sim_pos)), list(set(DMI_sim_neg)),\
               list(set(DMI_un_pos)), list(set(DMI_un_neg)),\
               list(set(N34_un_pos)), list(set(N34_un_neg)),\
               list(DMI_pos), list(DMI_neg), \
               list(N34_pos), list(N34_neg)


def PlotComp(comp, comp_var, title='Fig', fase=None, name_fig='Fig',
             save=False, dpi=200, levels=np.linspace(-1.5, 1.5, 13),
             contour=False, cmap='RdBu_r', number_events='', season = '',
             waf=False, px=None, py=None, text=True, SA=False,
             two_variables = False, comp2=None, step = 1,
             levels2=np.linspace(-1.5, 1.5, 13), contour0 = False):

    from numpy import ma
    import matplotlib.pyplot as plt


    if SA:
        fig = plt.figure(figsize=(5, 6), dpi=dpi)
    else:
        fig = plt.figure(figsize=(7, 3.5), dpi=dpi)

    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    crs_latlon = ccrs.PlateCarree()
    if SA:
        ax.set_extent([270,330, -60,20],crs_latlon)
    else:
        ax.set_extent([0, 359, -80, 40], crs=crs_latlon)


    im = ax.contourf(comp.lon[::step], comp.lat[::step], comp_var[::step,::step],
                     levels=levels,transform=crs_latlon, cmap=cmap, extend='both')
    if contour:
        values = ax.contour(comp.lon, comp.lat, comp_var, levels=levels,
                            transform=crs_latlon, colors='darkgray', linewidths=1)
        ax.clabel(values, inline=1, fontsize=5, fmt='%1.1f')

    if contour0:
        values = ax.contour(comp.lon, comp.lat, comp_var, levels=0,
                            transform=crs_latlon, colors='magenta', linewidths=1)
        ax.clabel(values, inline=1.5, fontsize=5, fmt='%1.1f')

    if two_variables:
        print('Plot Two Variables')
        comp_var2 = comp2['var'] ######## CORREGIR en caso de generalizar #############
        values2 = ax.contour(comp2.lon, comp2.lat, comp_var2, levels=levels2,
                            transform=crs_latlon, colors='k', linewidths=1)
        #ax.clabel(values2, inline=1, fontsize=5, fmt='%1.1f')


    cb = plt.colorbar(im, fraction=0.042, pad=0.035,shrink=0.8)
    cb.ax.tick_params(labelsize=8)
    ax.add_feature(cartopy.feature.LAND, facecolor='#d9d9d9')
    ax.add_feature(cartopy.feature.COASTLINE)
    # ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
    if SA:
        ax.set_xticks(np.arange(270, 330, 10), crs=crs_latlon)
        ax.set_yticks(np.arange(-60, 40, 20), crs=crs_latlon)
    else:
        ax.set_xticks(np.arange(30, 330, 60), crs=crs_latlon)
        ax.set_yticks(np.arange(-80, 20, 20), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=7)

    if waf:
        Q60 = np.percentile(np.sqrt(np.add(np.power(px, 2), np.power(py, 2))), 0)
        M = np.sqrt(np.add(np.power(px, 2), np.power(py, 2))) < Q60
        # mask array
        px_mask = ma.array(px, mask=M)
        py_mask = ma.array(py, mask=M)
        # plot vectors
        lons, lats = np.meshgrid(comp.lon.values, comp.lat.values)
        ax.quiver(lons[::20, ::20], lats[::20, ::20], px_mask[0, ::20, ::20],
                  py_mask[0, ::20, ::20], transform=crs_latlon,pivot='mid'
                  , scale=1/50)#, width=1.5e-3, headwidth=3.1,  # headwidht (default3)
                  #headlength=2.2)  # (default5))

    plt.title(str(title) + ' - ' + str(season) + '  ' + str(fase.split(' ', 1)[1]), fontsize=10)
    if text:
        plt.figtext(0.5, 0.01, number_events, ha="center", fontsize=10,
                bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5})
    plt.tight_layout()

    if save:
        plt.savefig(name_fig + str(season) + '_' + str(fase.split(' ', 1)[1]) + '.jpg')
        plt.close()
    else:
        plt.show()


def Plots(data, variable='var', neutral=None, DMI_pos=None, DMI_neg=None,
          N34_pos=None, N34_neg=None, neutral_name='', cmap='RdBu_r',
          dpi=200, mode='', levels=np.linspace(-1.5, 1.5, 13),
          name_fig='', save=False, contour=False, title="", waf=False,
          two_variables=False, data2=None, neutral2=None, levels2=None,
          season=None, text=True, SA=False, contour0=False, step=1,
          px=None, py=None):

    if two_variables == False:
        if data is None:
            if data2 is None:
                print('data None!')
            else:
                data = data2
                print('Data is None!')
                print('Using data2 instead data')
                levels = levels2
                neutral = neutral2

    def Title(DMI_phase, N34_phase, title=title):
        DMI_phase = set(DMI_phase)
        N34_phase = set(N34_phase)
        if mode.split(' ', 1)[0] != 'Simultaneus':
            if mode.split(' ', 1)[1] == 'IODs':
                title = title + mode + ': ' + str(len(DMI_phase)) + '\n' + 'against ' + clim
                number_events = str(DMI_phase)
            else:
                title = title + mode + ': ' + str(len(N34_phase)) + '\n' + 'against ' + clim
                number_events = str(N34_phase)

        elif mode.split(' ', 1)[0] == 'Simultaneus':
            title = title +mode + '\n' + 'IODs: ' + str(len(DMI_phase)) + \
                    ' - ENSOs: ' + str(len(N34_phase)) + '\n' + 'against ' + clim
            number_events = str(N34_phase)
        return title, number_events




    if data[0] != 0:
        comp = data[0] - neutral
        clim = neutral_name
        try:
            comp2 = data2[0] - neutral2
        except:
            comp2 = None
            print('One Variable')

        PlotComp(comp=comp, comp_var=comp[variable],
                 title=Title(DMI_phase=DMI_pos, N34_phase=N34_pos)[0],
                 fase=' - Positive', name_fig=name_fig,
                 save=save, dpi=dpi, levels=levels,
                 contour=contour, cmap=cmap,
                 number_events=Title(DMI_phase=DMI_pos, N34_phase=N34_pos)[1],
                 season=season,
                 waf=waf, px=px, py=py, text=text, SA=SA,
                 two_variables=two_variables,
                 comp2=comp2, step=step,
                 levels2=levels2, contour0=contour0)

    if data[1] != 0:
        comp = data[1] - neutral
        clim = neutral_name
        try:
            comp2 = data2[1] - neutral2
        except:
            comp2 = None
            print('One Variable')

        PlotComp(comp=comp, comp_var=comp[variable],
                 title=Title(DMI_phase=DMI_neg, N34_phase=N34_neg)[0],
                 fase=' - Negative', name_fig=name_fig,
                 save=save, dpi=dpi, levels=levels,
                 contour=contour, cmap=cmap,
                 number_events=Title(DMI_phase=DMI_neg, N34_phase=N34_neg)[1],
                 season=season,
                 waf=waf, px=px, py=py, text=text, SA=SA,
                 two_variables=two_variables,
                 comp2=comp2, step=step,
                 levels2=levels2, contour0=contour0)


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


    def xrFieldTimeDetrend(xrda, dim, deg=1):
        # detrend along a single dimension
        aux = xrda.polyfit(dim=dim, deg=deg)
        try:
            trend = xr.polyval(xrda[dim], aux.var_polyfit_coefficients[0])
        except:
            trend = xr.polyval(xrda[dim], aux.polyfit_coefficients[0])

        dt = xrda - trend
        return dt

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
        pp_20cr = xrFieldTimeDetrend(pp_20cr, 'time')

        return pp_20cr
    elif name == 'pp_gpcc':
        # GPCC2018
        aux = xr.open_dataset(pwd_datos + 'pp_gpcc.nc')
        # interpolado igual que 20cr, los dos son 1x1 pero con distinta grilla
        pp_gpcc = aux.sel(lon=slice(270, 330), lat=slice(20, -50))
        if interp:
            pp_gpcc = aux.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
        pp_gpcc = pp_gpcc.rename({'precip': 'var'})
        pp_gpcc = xrFieldTimeDetrend(pp_gpcc, 'time')

        return pp_gpcc
    elif name == 'pp_PREC':
        # PREC
        aux = xr.open_dataset(pwd_datos + 'pp_PREC.nc')
        pp_prec = aux.sel(lon=slice(270, 330), lat=slice(20, -60))
        if interp:
            pp_prec = pp_prec.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
        pp_prec = pp_prec.rename({'precip': 'var'})
        pp_prec = pp_prec.__mul__(365 / 12)  # mm/day -> mm/month
        pp_prec = xrFieldTimeDetrend(pp_prec, 'time')

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
        pp_ch = xrFieldTimeDetrend(pp_ch, 'time')

        return pp_ch
    elif name == 'pp_CMAP':
        # CMAP
        aux = xr.open_dataset(pwd_datos + 'pp_CMAP.nc')
        aux = aux.rename({'precip': 'var'})
        aux = aux.sel(lon=slice(270, 330), lat=slice(20, -50))
        if interp:
            pp_cmap = aux.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
        pp_cmap = aux.__mul__(365 / 12)  # mm/day -> mm/month
        pp_cmap = xrFieldTimeDetrend(pp_cmap, 'time')

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
        pp_gpcp = xrFieldTimeDetrend(pp_gpcp, 'time')

        return pp_gpcp
    elif name == 't_20CR-V3':
        # 20CR-v3
        aux = xr.open_dataset(pwd_datos + 't_20CR-V3.nc')
        t_20cr = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))
        t_20cr = t_20cr.rename({'air': 'var'})
        t_20cr = t_20cr - 273
        t_20cr = t_20cr.drop('time_bnds')
        t_20cr = xrFieldTimeDetrend(t_20cr, 'time')
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
        t_cru = xrFieldTimeDetrend(t_cru, 'time')
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
        t_beic = xrFieldTimeDetrend(t_beic, 'time')
        return t_beic

    elif name == 't_ghcn_cams':
        # GHCN

        aux = xr.open_dataset(pwd_datos + 't_ghcn_cams.nc')
        aux = aux.sel(lon=slice(270, 330), lat=slice(20, -60))
        if interp:
            aux = aux.interp(lon=t_20cr.lon.values, lat=t_20cr.lat.values)
        t_ghcn = aux.rename({'air': 'var'})
        t_ghcn = t_ghcn - 273
        t_ghcn = xrFieldTimeDetrend(t_ghcn, 'time')
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
        t_had = xrFieldTimeDetrend(t_had, 'time')

        return t_had

    elif name == 't_era20c':

        # ERA-20C
        aux = xr.open_dataset(pwd_datos + 't_era20c.nc')
        aux = aux.rename({'t2m': 'var', 'latitude': 'lat', 'longitude': 'lon'})
        aux = aux.sel(lon=slice(270, 330), lat=slice(20, -60))
        if interp:
            aux = aux.interp(lon=t_20cr.lon.values, lat=t_20cr.lat.values)
        t_era20 = aux - 273
        t_era20 = xrFieldTimeDetrend(t_era20, 'time')

        return t_era20
    elif name == 'pp_lieb':
        aux = xr.open_dataset(pwd_datos + 'pp_liebmann.nc')
        aux = aux.sel(time=slice('1985-01-01', '2010-12-31'))
        aux = aux.resample(time='1M', skipna=True).mean()
        aux = ChangeLons(aux, 'lon')
        pp_lieb = aux.sel(lon=slice(275, 330), lat=slice(-50, 20))
        pp_lieb = pp_lieb.__mul__(365 / 12)
        pp_lieb = pp_lieb.drop('count')
        pp_lieb = pp_lieb.rename({'precip': 'var'})
        pp_lieb = xrFieldTimeDetrend(pp_lieb, 'time')
        return pp_lieb


def xrFieldTimeDetrend(xrda, dim, deg=1):
    # detrend along a single dimension
    aux = xrda.polyfit(dim=dim, deg=deg)
    try:
        trend = xr.polyval(xrda[dim], aux.var_polyfit_coefficients[0])
    except:
        trend = xr.polyval(xrda[dim], aux.polyfit_coefficients[0])

    dt = xrda - trend
    return dt
