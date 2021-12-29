import xarray as xr
import numpy as np

dir = '/datos/luciano.andrian/ncfiles/nc_composites_dates/'
start = ('1920', '1950', '1980')
seasons = ("Full_Season", 'MJA', 'JJA', 'JAS', 'ASO', 'SON')

for i in start:
    for s in seasons:
        aux = xr.open_dataset(dir + 'Composite_' + i + '_2020_' + s + '.nc')
        neutro = aux.Neutral

        sim_pos = aux.DMI_sim_pos
        sim_neg = aux.DMI_sim_neg

        dmi_un_pos = aux.DMI_un_pos
        dmi_un_neg = aux.DMI_un_neg
        dmi_pos = aux.DMI_pos
        dmi_neg = aux.DMI_neg

        n34_un_pos = aux.N34_un_pos
        n34_un_neg = aux.N34_un_neg
        n34_pos = aux.N34_pos
        n34_neg = aux.N34_neg

        neutro_sim_pos = np.concatenate([neutro.values, sim_pos])

        del aux


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


data_to_concat = aux.N34_un_pos.values

import math

def PermuDatesComposite(data, data_to_concat, neutro, M=10, seed=42, mmin=-99, mmax=-99, name='Comp'):
    total = len(data_to_concat) + len(neutro)
    len1 = len(neutro)
    len2 = len(data_to_concat)

    total_perts = math.factorial(total)/(math.factorial(len2)*math.factorial(len1))
    #
    # if total_perts > 10000:
    #     M = 10000
    #     print('M = 10000')
    # else:
    #     M = total_perts
    #     print('M = ' + str(total_perts))


    print('Seed ' + str(seed))

    if data_to_concat[0] != 0:
        neutro_concat = np.concatenate([neutro, data_to_concat])

        rn = np.random.RandomState(seed)
        for i in range(M):
            dates_rn = rn.permutation(neutro_concat)
            neutro_new = dates_rn[0:len(neutro)]
            data_new = dates_rn[len(neutro):]

            neutro_comp = CompositeSimple(original_data=data, index=neutro_new,
                                          mmin=mmin, mmax=mmax)
            data_comp = CompositeSimple(original_data=data, index=data_new,
                                        mmin=mmin, mmax=mmax)

            comp = data_comp - neutro_comp
            comp.to_netcdf('/datos/luciano.andrian/ncfiles/nc_comps/' + name + str(i) + '_' + season + '.nc')

import multiprocessing as mp
pool = mp.Pool(mp.cpu_count() - 10)
[pool.apply_async(PermuDatesComposite, args=(data, data_to_concat, neutro, m, 42, 9, 11)) for m in [100]]
pool.close()



###########################################

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

w_dir = '/home/luciano.andrian/doc/salidas/'
out_dir = '/home/luciano.andrian/doc/salidas/ENSO_IOD/composite/'
pwd = '/datos/luciano.andrian/ncfiles/'

import multiprocessing as mp

########################################################################################################################
variables = 'hgt200'
variables_t_pp = ['pp_20CR-V3', 'pp_gpcc', 'pp_PREC', 'pp_chirps', 'pp_CMAP',
                  'pp_gpcp', 't_20CR-V3', 't_cru', 't_BEIC', 't_ghcn_cams', 't_hadcrut', 't_era20c']
interp = [False, False, False, False, False, False, False, False, False, False, False, False]

seasons = [7, 9, 10]
seasons_name = ['JJA', 'JAS', 'ASO', 'SON']

two_variables = True
contour = True
SA = True
step = 1
contour0 = False

scales_pp_t = [np.linspace(-30, 30, 13),  # pp
               np.linspace(-1, 1, 21)]  # t
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

data = xr.open_dataset(pwd + variables + '.nc')
print('data open:' + variables + '.nc')
data = data.sel(time=slice(i + '-01-01', '2020-12-01'), lat=slice(20, -90))











