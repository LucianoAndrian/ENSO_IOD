########################################################################################################################
"""
Igual que ENSO-IOD.py pero sólo prepara las fechas (años) de los eventos, sim, un y all. NO COMPOSITE
"""
########################################################################################################################
import xarray as xr
import pandas as pd
pd.options.mode.chained_assignment = None
import os
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from ENSO_IOD_Funciones import Nino34CPC
from ENSO_IOD_Funciones import DMI
from ENSO_IOD_Funciones import MultipleComposite

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

w_dir = '/home/luciano.andrian/doc/salidas/'
out_dir = '/home/luciano.andrian/doc/salidas/ENSO_IOD/composite/'
pwd = '/datos/luciano.andrian/ncfiles/'



########################################################################################################################

seasons = [6, 7, 8, 9, 10]
seasons_name = ['MJJ', 'JJA', 'JAS','ASO', 'SON']

full_season2 = [True, False]
bwa = False

# Usando T: Beic y PP:GPCC se cubre 1920-2020
# Se puede usar junto los ERA-Frankenstein... o solo ERA5 o ERA5 + ERA5_50-78

start = [1920,1950,1980] #*ERA5 va desde 1979 pero es una molestia en Nino34CPC y su climatologia movil.
end = 2020
########################################################################################################################

for i in start:
    for fs in full_season2:

        dmi, aux, dmi_aux = DMI(filter_bwa=False, start_per=i, end_per=end)
        del aux
        aux = xr.open_dataset("/pikachu/datos4/Obs/sst/sst.mnmean_2020.nc")
        n34 = Nino34CPC(aux, start=i, end=end)[2]

        if fs:
            full_season = True
            s = 666
            print('Full Season')
            Neutral, DMI_sim_pos, DMI_sim_neg, DMI_un_pos, DMI_un_neg, N34_un_pos, N34_un_neg, DMI_pos, \
            DMI_neg, N34_pos, N34_neg = MultipleComposite(var=dmi_aux, n34=n34, dmi=dmi, season=s, start=i,
                                                          full_season=full_season, compute_composite=False)

            ds = xr.Dataset(
                data_vars={
                    'Neutral': (Neutral),
                    "DMI_sim_pos": (DMI_sim_pos),
                    "DMI_sim_neg": (DMI_sim_neg),
                    "DMI_un_pos": (DMI_un_pos),
                    "DMI_un_neg": (DMI_un_neg),
                    "N34_un_pos": (N34_un_pos),
                    "N34_un_neg": (N34_un_neg),
                    "DMI_pos": (DMI_pos),
                    "DMI_neg": (DMI_neg),
                    "N34_pos": (N34_pos),
                    "N34_neg": (N34_neg),
                }
            )
            ds.to_netcdf(pwd + 'nc_composites_dates/' + 'Composite_' +
                         str(i) + '_' + str(end) + '_Full_Season.nc')

        else:
            full_season = False
            for s in seasons:
                print(seasons_name[s-6])
                Neutral, DMI_sim_pos, DMI_sim_neg, DMI_un_pos, DMI_un_neg, N34_un_pos, N34_un_neg, DMI_pos, \
                DMI_neg, N34_pos, N34_neg = MultipleComposite(var=dmi_aux, n34=n34, dmi=dmi, season=s-1, start=i,
                                                              full_season=full_season, compute_composite=False)

                ds = xr.Dataset(
                    data_vars={
                        'Neutral':(Neutral),
                        "DMI_sim_pos": (DMI_sim_pos),
                        "DMI_sim_neg": (DMI_sim_neg),
                        "DMI_un_pos": (DMI_un_pos),
                        "DMI_un_neg": (DMI_un_neg),
                        "N34_un_pos": (N34_un_pos),
                        "N34_un_neg": (N34_un_neg),
                        "DMI_pos": (DMI_pos),
                        "DMI_neg": (DMI_neg),
                        "N34_pos": (N34_pos),
                        "N34_neg": (N34_neg),
                    }
                )
                ds.to_netcdf(pwd + 'nc_composites_dates/' + 'Composite_' +
                             str(i) + '_' + str(end) + '_' + seasons_name[s-6] +'.nc')