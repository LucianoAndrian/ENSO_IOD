import time
import xarray as xr
import numpy as np
from matplotlib import colors
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
########################################################################################################################
cases_dir = '/pikachu/datos/luciano.andrian/cases/'
out_dir = '/home/luciano.andrian/doc/salidas/ENSO_IOD/Modelos/SNR_CFSv2/'
save=True
dpi=200
# Funciones ############################################################################################################
def CompositeSimpleCFSv2_r(original_data, index, l, name_case):
    """
    A diferencia del los demás, el promedio trimestral ya está hecho.

    """

    if len(index) != 0:
        comp_field = original_data.sel(Year=index)
        dims= comp_field['var'].shape
        r = 24
        y = dims[0]
        num_case = r * y
        if len(dims)>4:
            l = dims[1]
            num_case *=  l


        if len(comp_field.Year) != 0:
            if l > 1:
                comp_field = comp_field.mean(['r', name_case, 'l'], skipna=True)
            else:
                comp_field = comp_field.mean(['r', name_case], skipna=True)
        else:  # si sólo hay un año
            comp_field = comp_field.mean(['r'])

            comp_field = comp_field.drop_dims(['Year'])
            if l > 1:
                comp_field = comp_field.mean(['l'])


        return comp_field, num_case
    else:
        print(' len index = 0')

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
                     levels=levels, transform=crs_latlon, cmap=cmap, extend='both')
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
    from ENSO_IOD_Funciones import ChangeLons

    if name == 'pp_gpcc':
        # GPCC2018
        aux = xr.open_dataset('/datos/luciano.andrian/ncfiles/' + 'pp_gpcc.nc')
        pp_gpcc = aux.sel(lon=slice(270, 330), lat=slice(15, -60))
        if interp:
            pp_gpcc = aux.interp(lon=lon_interp, lat=lat_interp)
        pp_gpcc = pp_gpcc.rename({'precip': 'var'})
        pp_gpcc = pp_gpcc.sel(time=slice('1982-01-01','2020-12-01'))

        return pp_gpcc
########################################################################################################################
variables = ('prec', 'tref')
seasons = ('JJA', 'JAS', 'ASO', 'SON')
conj_name= ('0','1','3','7', 'todo')

cases = ['DMI_sim_pos', 'DMI_sim_neg', 'DMI_neg', 'DMI_pos', 'DMI_un_pos', 'DMI_un_neg',
         'N34_pos', 'N34_neg', 'N34_un_pos', 'N34_un_neg']

title_case = ['DMI-ENSO simultaneous positive phase ',
              'DMI-ENSO simultaneous negative phase ',
              'DMI negative phase ',
              'DMI positive phase ',
              'DMI isolated positive phase ',
              'DMI isolated negative phase ',
              'ENSO positive phase ',
              'ENSO negative phase ',
              'ENSO isolated positive phase ',
              'ENSO isolated negative phase ']

scale = [np.linspace(-30,30,13), np.linspace(-1,1,13)]
scale_sd = [np.linspace(0,70,9), np.linspace(0,1,11)]

cbar_pp = colors.ListedColormap(['#003C30', '#004C42', '#0C7169', '#79C8BC', '#B4E2DB',
                                'white',
                                '#F1DFB3', '#DCBC75', '#995D13', '#6A3D07', '#543005', ][::-1])
cbar_pp.set_under('#3F2404')
cbar_pp.set_over('#00221A')
cbar_pp.set_bad(color='white')

cbar_t = colors.ListedColormap(['#B9391B', '#CD4838', '#E25E55', '#F28C89', '#FFCECC',
                              'white',
                              '#B3DBFF', '#83B9EB', '#5E9AD7', '#3C7DC3', '#2064AF'][::-1])
cbar_t.set_over('#9B1C00')
cbar_t.set_under('#014A9B')
cbar_t.set_bad(color='white')

colormap= [cbar_pp, cbar_t]
colormap_sd = ['PuBuGn', 'YlOrRd']

cbar_snr = colors.ListedColormap(['#FFFFFF', '#FDE7A9', '#FEB77E', '#FB8761', '#CA3E72', '#A1307E',
                              '#782281',
                              '#4F127B', '#251255'])
cbar_snr.set_over('#251255')
cbar_snr.set_under('#FFFFFF')
cbar_snr.set_bad(color='white')

mask = OpenDataSet('pp_gpcc', interp=True,
                   lat_interp=np.linspace(-60,15,76),
                   lon_interp=np.linspace(275,330,56))
mask = mask.mean('time')
mask = xr.where(np.isnan(mask), mask, 1)

scales = [np.linspace(-30, 30, 13), np.linspace(-1.2,1.2,13)]
scales_sd = [np.linspace(0,50,11), np.linspace(0,1,11)]
scales_snr = [np.linspace(0.1,1,10),np.linspace(0.1,1,10)]

colormap = [cbar_pp,cbar_t]
colormap_sd = ['PuBuGn', 'YlOrRd']

# v = 'prec'
# s='SON'
# c='DMI_un_neg'
# cj= '7'
v_count = 0
for v in variables:
    for s in seasons:
        print(s)
        for cj in conj_name:
            print(cj)
            c_count = 0
            for c in cases:
                print(c)
                data_neutral = xr.open_dataset(cases_dir + v + '_' + s + '_Neutral_' + cj + '.nc')
                data_case = xr.open_dataset(cases_dir + v + '_' + s + '_' + c + '_' + cj + '.nc')

                num_case = len(data_case.case)
                data_neutral['lon'] = np.linspace(275,330,56)
                data_neutral['lat'] = np.linspace(-60,15,76)

                data_case['lon'] = np.linspace(275,330,56)
                data_case['lat'] = np.linspace(-60,15,76)

                # signal
                aux = data_case.mean('case') - data_neutral.mean('case')
                aux *= mask
                Plot(aux, levels=scales[v_count], cmap=colormap[v_count],
                     dpi=dpi, step=1,
                     name_fig=v + '_set_' + cj + '_' + c + '_' + s + '.nc',
                     title='Mean Composite - CFSv2 - ' + s + '\n' + title_case[c_count] + '\n' + ' ' + v + ' - ' +
                           'Leads: ' + str(cj) + ' - ' + 'Casos: ' + str(num_case),
                     save=save)

                # spread
                aux2 = data_case.std('case')
                aux2 *= mask
                Plot(aux2, levels=scales_sd[v_count], cmap=colormap_sd[v_count],
                     dpi=dpi, step=1,
                     name_fig=v + 'SD_set_' + cj + '_' + c + '_' + s + '.nc',
                     title='SD Composite - CFSv2 - ' + s + '\n' + title_case[c_count] + '\n' + ' ' + v + ' - ' +
                           'Leads: ' + str(cj) + ' - ' + 'Casos: ' + str(num_case),
                     save=save)

                # SNR
                snr = aux.__abs__() ** 2 / aux2 ** 2
                snr *=mask
                Plot(snr, levels=scales_snr[v_count], cmap=cbar_snr,
                     dpi=dpi, step=1,
                     name_fig=v + 'SNR_set_' + cj + '_' + c + '_' + s + '.nc',
                     title='SNR Composite - CFSv2 - ' + s + '\n' + title_case[c_count] + '\n' + ' ' + v + ' - ' +
                           'Leads: ' + str(cj) + ' - ' + 'Casos: ' + str(num_case),
                     save=save)
                c_count += 1
    v_count += 1

