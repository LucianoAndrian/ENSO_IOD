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
from ENSO_IOD_Funciones import OpenDatasets

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




