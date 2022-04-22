import xarray as xr
import numpy as np
from matplotlib import colors
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import warnings
#warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
warnings.filterwarnings('ignore')
########################################################################################################################
cases_dir = '/pikachu/datos/luciano.andrian/cases/'
out_dir = '/home/luciano.andrian/doc/salidas/ENSO_IOD/Modelos/SpatialProb/'
save=True
dpi=200
# Funciones ############################################################################################################
def Plot(comp, levels = np.linspace(-1,1,11), cmap='RdBu',
         dpi=100, save=True, step=1,
         name_fig='fig', title='title'):


    import matplotlib.pyplot as plt

    comp_var = comp['var']
    fig = plt.figure(figsize=(5, 6), dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    crs_latlon = ccrs.PlateCarree()
    ax.set_extent([270,330, -60,20], crs_latlon)

    im = ax.contourf(comp.lon[::step], comp.lat[::step], comp_var[::step, ::step],
                     levels=levels, transform=crs_latlon, cmap=cmap)
    cb = plt.colorbar(im, fraction=0.042, pad=0.035,shrink=0.8)
    cb.ax.tick_params(labelsize=8)
    #ax.add_feature(cartopy.feature.LAND, facecolor='#d9d9d9')
    ax.add_feature(cartopy.feature.LAND, facecolor='lightgrey')
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
    ax.set_xticks(np.arange(270, 330, 10), crs=crs_latlon)
    ax.set_yticks(np.arange(-60, 40, 20), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=7)

    plt.title(title, fontsize=10)
    plt.tight_layout()

    if save:
        plt.savefig(out_dir + name_fig + '.jpg')
        plt.close()
    else:
        plt.show()

def OpenDataSet(name, interp=False, lat_interp=None, lon_interp=None):

    if name == 'pp_gpcc':
        # GPCC2018
        aux = xr.open_dataset('/datos/luciano.andrian/ncfiles/' + 'pp_gpcc.nc')
        pp_gpcc = aux.sel(lon=slice(270, 330), lat=slice(15, -60))
        if interp:
            pp_gpcc = aux.interp(lon=lon_interp, lat=lat_interp)
        pp_gpcc = pp_gpcc.rename({'precip': 'var'})
        pp_gpcc = pp_gpcc.sel(time=slice('1982-01-01','2020-12-01'))

        return pp_gpcc

def SpatialProbability(data, mask):
    prob = xr.where(np.isnan(mask), mask, 1)
    for ln in range(0, 56):
        for lt in range(0, 76):
            prob['var'][lt, ln] = \
                len(data['var'][:, lt, ln][~np.isnan(data['var'][:, lt, ln].values)].values) \
                / len(data['var'][:, lt, ln])
    return prob*mask
########################################################################################################################
variables = ('prec', 'tref')
seasons = ('JJA', 'JAS', 'ASO', 'SON')
set_name= ('0','1','3','7')

cases = ['sim_pos', 'sim_neg', 'DMI_neg', 'DMI_pos', 'DMI_un_pos', 'DMI_un_neg',
         'N34_pos', 'N34_neg', 'N34_un_pos', 'N34_un_neg', 'sim_DMIneg_N34pos', 'sim_DMIpos_N34neg']

title_case = ['DMI-ENSO simultaneous positive phase ',
              'DMI-ENSO simultaneous negative phase ',
              'DMI negative phase ',
              'DMI positive phase ',
              'DMI isolated positive phase ',
              'DMI isolated negative phase ',
              'ENSO positive phase ',
              'ENSO negative phase ',
              'ENSO isolated positive phase ',
              'ENSO isolated negative phase ',
              'DMI negative and ENSO positive',
              'DMI positive and ENSO negative']

mask = OpenDataSet('pp_gpcc', interp=True,
                   lat_interp=np.linspace(-60,15,76),
                   lon_interp=np.linspace(275,330,56))
mask = mask.mean('time')
mask = xr.where(np.isnan(mask), mask, 1)

scales = [np.linspace(-30, 30, 13), np.linspace(-1.2,1.2,13)]
scales_sd = [np.linspace(0,50,11), np.linspace(0,1,11)]
scales_snr = [np.linspace(0.1,1,10),np.linspace(0.1,1,10)]
#
# v = 'prec'
# s='JJA'
# c='DMI_un_neg'
# s_n= '0'
v_count = 0
for v in variables:
    if v == 'prec':
        fix_factor = 30
    else:
        fix_factor=1
    for s in seasons:
        #print(s)
        for s_n in set_name:
            #print(s_n)
            c_count = 0
            for c in cases:
                #print(c)
                data_neutral = xr.open_dataset(cases_dir + v + '_' + s +'_Set' + s_n + '_NEUTRO.nc').drop(['L', 'r'])
                data_neutral = data_neutral.rename({v:'var'})
                data_neutral *= fix_factor
                try:
                    data_case = xr.open_dataset(cases_dir + v + '_' + s +'_Set' + s_n + '_' + c + '.nc').drop(['L', 'r'])
                    data_case *= fix_factor
                    data_case = data_case.rename({v: 'var'})
                    data_case -= data_neutral.mean('time')
                    aux = data_case.std('time')  # - data_neutral.mean('time')
                    aux *= mask

                    # <-1*std
                    aux2 = xr.where(data_case<-1*aux, data_case, np.nan)
                    prob = SpatialProbability(aux2, mask)
                    Plot(prob, levels=np.linspace(0,1,11), cmap='Spectral_r',
                         dpi=dpi, step=1,
                         name_fig=v + '_set_' + s_n + '_-1SD_' + c + '_' + s + '.nc',
                         title='Spatial Probability - CFSv2 - ' + s + '\n' + title_case[c_count] + '\n' + ' ' +
                               v + '<-1*std' + ' - ' + 'Leads: ' + s_n, save=save)

                    # <0
                    aux2 = xr.where(data_case<0, data_case, np.nan)
                    prob = SpatialProbability(aux2, mask)
                    Plot(prob, levels=np.linspace(0,1,11), cmap='Spectral_r',
                         dpi=dpi, step=1,
                         name_fig=v + '_set_' + s_n + '_-0SD_' + c + '_' + s + '.nc',
                         title='Spatial Probability - CFSv2 - ' + s + '\n' + title_case[c_count] + '\n' + ' ' +
                               v + '<0' + ' - ' + 'Leads: ' + s_n, save=save)

                    # >1*std
                    aux2 = xr.where(data_case>1*aux, data_case, np.nan)
                    prob = SpatialProbability(aux2, mask)
                    Plot(prob, levels=np.linspace(0,1,11), cmap='Spectral_r',
                         dpi=dpi, step=1,
                         name_fig=v + '_set_' + s_n + '_+1SD_' + c + '_' + s + '.nc',
                         title='Spatial Probability - CFSv2 - ' + s + '\n' + title_case[c_count] + '\n' + ' ' +
                               v + '>1*std' + ' - ' + 'Leads: ' + s_n, save=save)

                    # >2*std
                    aux2 = xr.where(data_case>2*aux, data_case, np.nan)
                    prob = SpatialProbability(aux2, mask)
                    Plot(prob, levels=np.linspace(0,1,11), cmap='Spectral_r',
                         dpi=dpi, step=1,
                         name_fig=v + '_set_' + s_n + '_+2SD_' + c + '_' + s + '.nc',
                         title='Spatial Probability - CFSv2 - ' + s + '\n' + title_case[c_count] + '\n' + ' ' +
                               v + '>2*std' + ' - ' + 'Leads: ' + s_n, save=save)
                except:
                    x = None
                    #print('No' + c + ' in Set' + s_n + ' at ' + s )

                c_count += 1
    v_count += 1