import time

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
########################################################################################################################
nc_date_dir = '/datos/luciano.andrian/ncfiles/nc_composites_dates/'
data_dir = '/datos/luciano.andrian/ncfiles/'
out_dir = '/home/luciano.andrian/doc/salidas/ENSO_IOD/composite/diff/'
sig_dir = '/datos/luciano.andrian/ncfiles/nc_quantiles/'

start = ('1920', '1950')
seasons = ("Full_Season", 'JJA', 'ASO', 'SON')
min_max_months = [[7,11], [6,8],[8,10],[9,11]]
#variables = ['hgt200', 'div', 'psl', 'sf', 'vp', 't_cru', 't_BEIC', 'pp_gpcc']
variables = ['hgt200','psl','vp', 't_cru', 'pp_gpcc']


cases = ['DMI_sim_pos', 'DMI_sim_neg', 'DMI_neg', 'DMI_pos', 'DMI_un_pos', 'DMI_un_neg',
         'N34_pos', 'N34_neg', 'N34_un_pos', 'N34_un_neg']

# scales = [np.linspace(-450, 450, 21),  #hgt
#           np.linspace(-0.45e-5, 0.45e-5, 13),  # div
#           np.linspace(-3, 3, 13),  #psl
#           np.linspace(-4.5e6, 4.5e6, 13),  #sf
#           np.linspace(-4.5e6, 4.5e6, 13),  #vp
#           np.linspace(-1, 1 ,17),  #t
#           np.linspace(-1, 1 ,17),  #t
#           np.linspace(-30, 30, 13)] #pp

scales = [np.linspace(-450, 450, 21),  #hgt
          np.linspace(-3, 3, 13),  #psl
          np.linspace(-4.5e6, 4.5e6, 13),  #vp
          np.linspace(-1, 1 ,17),  #t
          np.linspace(-30, 30, 13)] #pp

scales_cont = [np.linspace(-450, 450, 7),  #hgt
          np.linspace(-3, 3, 5),  #psl
          np.linspace(-4.5e6, 4.5e6, 5),  #vp
          np.linspace(-1, 1 ,17),  #t
          np.linspace(-30, 30, 13)] #pp



#SA = [False,False,False,False,False,True, True, True]
SA = [False,False,False, True, True]
#step = [1,6,1,1,1,1,1,1]
step = [1,1,1,1,1,1,1,1]
#text = True

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

v_name = ['HGT 200hPa', 'Divergence', 'PSL',
          'Stream Function', 'Potential Velocity',
          'Temperature - Cru', 'Temperature - BEIC', 'Precipitation - GPCC']

v_name = ['HGT 200hPa', 'PSL', 'Potential Velocity',
          'Temperature - Cru', 'Precipitation - GPCC']


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

cmap = [cbar, cbar, cbar, cbar, 'BrBG']

contours=[True, True, True, False, False]

## Functions ###########################################################################################################
def CompositeSimple(original_data, index, mmin, mmax):
    def is_months(month, mmin, mmax):
        return (month >= mmin) & (month <= mmax)

    if len(index) != 0:
        comp_field = original_data.sel(time=original_data.time.dt.year.isin([index]))
        comp_field = comp_field.sel(
            time=is_months(month=comp_field['time.month'], mmin=mmin, mmax=mmax))
        if len(comp_field.time) != 0:
            comp_field = comp_field.mean(['time'], skipna=True)
        else:  # si sólo hay un año
            comp_field = comp_field.drop_dims(['time'])

        return comp_field
    else:
        print(' len index = 0')


def Plot(comp_diff, comp_sim, comp_n34_dmi,
         levels = np.linspace(-1,1,11), cmap='RdBu',
         SA=False, dpi=100, save=True, step=1,
         name_fig='fig', title='title', contours=True,
         levels_cont=np.linspace(-1,1,11)):

    from numpy import ma
    import matplotlib.pyplot as plt

    comp_diff_var = comp_diff['var']
    comp_sim_var = comp_sim['var']
    comp_n34_dmi_var = comp_n34_dmi['var']

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


    im = ax.contourf(comp_diff.lon[::step], comp_diff.lat[::step], comp_diff_var[::step,::step],
                     levels=levels,transform=crs_latlon, cmap=cmap, extend='both')
    if contours:
        ax.contour(comp_sim.lon, comp_sim.lat, comp_sim_var, levels=levels_cont,
                   transform=crs_latlon, colors='k', linewidths=1)

        # ax.contour(comp_n34_dmi.lon, comp_n34_dmi.lat, comp_n34_dmi_var, levels=levels_cont,
        #            transform=crs_latlon, colors='magenta', linewidths=1)

    cb = plt.colorbar(im, fraction=0.042, pad=0.035,shrink=0.8)
    cb.ax.tick_params(labelsize=8)
    ax.add_feature(cartopy.feature.LAND, facecolor='#d9d9d9')
    ax.add_feature(cartopy.feature.COASTLINE)
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

    plt.title(title, fontsize=10)
    # if text:
    #     plt.figtext(0.5, 0.01, 'Number of events: ' + str(number_events), ha="center", fontsize=10,
    #             bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5})
    plt.tight_layout()

    if save:
        plt.savefig(out_dir + name_fig + '.jpg')
        plt.close()
    else:
        plt.show()

########################################################################################################################
c_var=0
for v in variables:
    for i in start:
        print('Período: ' + i + '- 2020')
        print('Open ' + v + '.nc')

        if c_var >=3:
            print('using1x1')
            data = xr.open_dataset(data_dir + v + '1x1.nc')
        else:
            data = xr.open_dataset(data_dir + v + '.nc')

        if v == 'hgt200':
            print('drop level')
            data = data.drop('level')
        elif v == 'psl':
            print('to hPa')
            data = data.__mul__(1 / 100)

        count = 0
        for s in seasons:
            aux = xr.open_dataset(nc_date_dir + 'Composite_' + i + '_2020_' + s + '.nc')
            neutro = aux.Neutral
            dmi_un_pos = aux.DMI_un_pos
            dmi_un_neg = aux.DMI_un_neg

            n34_un_pos = aux.N34_un_pos
            n34_un_neg = aux.N34_un_neg

            dmi_n34_sim_pos = aux.DMI_sim_pos
            dmi_n34_sim_neg = aux.DMI_sim_neg

            aux.close()

            mmonth = min_max_months[count]
            print(s)
            print(mmonth)

            mmin = mmonth[0]
            mmax = mmonth[-1]

            neutro_comp = CompositeSimple(original_data=data, index=neutro,
                                          mmin=mmin, mmax=mmax)

            #positive phase
            if (dmi_n34_sim_pos.values[0] !=0) & (dmi_un_pos[0] != 0):
                comp_sim_pos = CompositeSimple(original_data=data, index=dmi_n34_sim_pos.values,
                                               mmin=mmin, mmax=mmax)
                comp_sim_pos = comp_sim_pos - neutro_comp

                comp_dmi_pos = CompositeSimple(original_data=data, index=dmi_un_pos.values,
                                               mmin=mmin, mmax=mmax)
                comp_dmi_pos = comp_dmi_pos - neutro_comp

                comp_n34_pos = CompositeSimple(original_data=data, index=n34_un_pos.values,
                                               mmin=mmin, mmax=mmax)
                comp_n34_pos = comp_n34_pos - neutro_comp

                n34_add_dmi = (comp_n34_pos + comp_dmi_pos)/(len(dmi_n34_sim_pos.values)+len(dmi_un_pos.values))


                Plot(comp_diff=n34_add_dmi, comp_sim=n34_add_dmi, comp_n34_dmi=n34_add_dmi,
                     cmap=cmap[c_var], levels=scales[c_var], levels_cont=scales[c_var]/2,
                     SA=SA[c_var], step=step[c_var], dpi=200, contours=True,
                     title=v_name[c_var] + ' Positive phase  - ' + s + '\n' + '(ENSO + IOD)',
                     name_fig='Comp_ENSO_add_IOD-' + v + '_' + s + '-POS_' + i + '_2020',
                     save=True)

                Plot(comp_diff=comp_sim_pos, comp_sim=comp_sim_pos, comp_n34_dmi=n34_add_dmi,
                     cmap=cmap[c_var], levels=scales[c_var], levels_cont=scales_cont[c_var],
                     SA=SA[c_var], step=step[c_var], dpi=200, contours=False,
                     title=v_name[c_var] + ' Positive phase - ' + s + '\n' + '(ENSO sim IOD)',
                     name_fig='Comp_SIM_ENSO-IOD-' + v + '_' + s + '-POS_' + i + '_2020',
                     save=True)


            # negative phase
            if (dmi_n34_sim_neg.values[0] !=0) & (dmi_un_neg.values[0] != 0):
                comp_sim_neg = CompositeSimple(original_data=data, index=dmi_n34_sim_neg.values,
                                               mmin=mmin, mmax=mmax)
                comp_sim_neg = comp_sim_neg - neutro_comp

                comp_dmi_neg = CompositeSimple(original_data=data, index=dmi_un_neg.values,
                                               mmin=mmin, mmax=mmax)
                comp_dmi_neg = comp_dmi_neg - neutro_comp

                comp_n34_neg = CompositeSimple(original_data=data, index=n34_un_neg.values,
                                               mmin=mmin, mmax=mmax)
                comp_n34_neg = comp_n34_neg - neutro_comp

                n34_add_dmi = (comp_n34_neg + comp_dmi_neg)/(len(dmi_n34_sim_neg.values)+len(dmi_un_neg.values))

                diff = n34_add_dmi - comp_sim_neg

                Plot(comp_diff=n34_add_dmi, comp_sim=n34_add_dmi, comp_n34_dmi=n34_add_dmi,
                     cmap=cmap[c_var], levels=scales[c_var], levels_cont=scales[c_var]/2,
                     SA=SA[c_var], step=step[c_var], dpi=200, contours=True,
                     title=v_name[c_var] + ' Negative phase - ' + s + '\n' + '(ENSO + IOD)',
                     name_fig='Comp_ENSO_add_IOD-' + v + '_' + s + '-NEG_' + i + '_2020',
                     save=True)

                Plot(comp_diff=comp_sim_neg, comp_sim=comp_sim_neg, comp_n34_dmi=n34_add_dmi,
                     cmap=cmap[c_var], levels=scales[c_var], levels_cont=scales_cont[c_var],
                     SA=SA[c_var], step=step[c_var], dpi=200, contours=False,
                     title=v_name[c_var] + ' Negative phase - ' + s + '\n' + '(ENSO sim IOD)',
                     name_fig='Comp_SIM_ENSO-IOD-' + v + '_' + s + '-NEG_' + i + '_2020',
                     save=True)

            count += 1

    c_var += 1



