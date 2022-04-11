"""
Selecciona los conjuntos de leads de NMME CFSv2 en JJA, JAS, ASO y SON en hindcast y realtime
"""
import numpy as np
import xarray as xr
from ENSO_IOD_Funciones import SelectNMMEFiles
########################################################################################################################
dir_hc = '/pikachu/datos/osman/nmme/monthly/hindcast/'
dir_rt = '/pikachu/datos/osman/nmme/monthly/real_time/'
out_dir = '/datos/luciano.andrian/ncfiles/NMME_CFSv2/'

variables = ['tref', 'prec']
anios = np.arange(1982, 2021)

leads = [0,1,2,3,4,5,6,7]
seasons = ['JJA', 'JAS', 'ASO', 'SON']
mmonth_seasons = [7,8,9,10]

for v in variables:
    print(v)

    for l in leads:
        print('Lead: ' + str(l))

        for m in mmonth_seasons:
            print('Forecast for ' + seasons[m - 7])
            in_month = str(m - l - 1) # VER!
            if in_month == '0':
                in_month = '12'
            elif in_month == '-1':
                in_month = '11'
            print('issued at ' + in_month)
            print('Loading years...')

            check=True # para no "mergear" en el primer paso
            for y in anios:

                #correccion en el año para leads>5 en JJA y JAS
                if in_month == '12' or in_month == '11':
                    y += 1

                #Selecciona los archivos (nombres)
                files = SelectNMMEFiles(model_name='NCEP-CFSv2', variable=v,
                                        dir=dir_hc, anio=str(y), in_month=in_month)
                for f in files:
                    data = xr.open_dataset(f, decode_times=False) #el formato de Leads, meses, etc.. no le gusta
                    leads_data = data.L
                    r = data.M.values
                    if r <= 24: #>24 nan
                        s = data.S.values
                        #Dominio y leads para una estacion [l:l+3]
                        data = data.sel(X=slice(275, 330), Y=slice(-60, 15), L=leads_data[l:l+3], S=s[0])
                        data = data.drop(['S'])
                        data = data.rename({'X': 'lon', 'Y': 'lat', 'M': 'r'})
                        data = data.mean('L')
                        # agrega año para identificar el evento y concatenar en esta dimencion
                        data = data.expand_dims({'Year':[y]})
                        if check:
                            data_final = data
                            check = False
                            print('check first file')
                        else:
                            data_final = xr.merge([data_final, data])

                # lo mismo para real-time (dir=dir_rt)
                # no es problema que esté dentro del mismo for ya que
                # si SelectNMMEfiles no encuentra el archivo f in files
                # no hace nada.
                files = SelectNMMEFiles(model_name='NCEP-CFSv2', variable=v,
                                        dir=dir_rt, anio=str(y), in_month=in_month)

                for f in files:
                    data = xr.open_dataset(f, decode_times=False)
                    leads_data = data.L
                    r = data.M.values
                    if r <= 24:
                        s = data.S.values
                        data = data.sel(X=slice(275, 330), Y=slice(-60, 15), L=leads_data[l:l+3], S=s[0])
                        data = data.drop(['S'])
                        data = data.rename({'X': 'lon', 'Y': 'lat', 'M': 'r'})
                        data = data.mean('L')
                        data = data.expand_dims({'Year':[y]})
                        data_final = xr.merge([data_final, data])

            print('save as: ' + v + '_CFSv2_' + seasons[m - 7] + '_Lead_' + str(l) + '.nc')
            data_final.to_netcdf(out_dir + v + '_CFSv2_' + seasons[m - 7] + '_Lead_' + str(l) + '.nc' )
            del data_final
            del data