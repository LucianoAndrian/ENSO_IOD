import xarray as xr
import numpy as np
import multiprocessing as mp
import os
import math
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# Functions ############################################################################################################

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


def NumberPerts(data_to_concat, neutro, num = 0):
    total = len(data_to_concat) + len(neutro)
    len1 = len(neutro)
    len2 = len(data_to_concat)

    total_perts = math.factorial(total) / (math.factorial(len2) * math.factorial(len1))

    if num == 0:

        if total_perts >= 10000:
            tot = 10000
            print('M = 10000')
        else:
            tot = total_perts
            print('M = ' + str(total_perts))

    else:
        tot = num

    jump = 9
    M = []
    n = 0

    while n < tot:
        #aux = list(np.linspace((0 + n), (n + jump*100), (jump + 1)))
        aux = list(np.linspace((0 + n), (n + jump), (jump + 1)))
        M.append(aux)
        #n = n + 1
        n = n + jump + 1

    return M

########################################################################################################################

nc_date_dir = '/datos/luciano.andrian/ncfiles/nc_composites_dates/'
data_dir = '/datos/luciano.andrian/ncfiles/'

start = ('1920', '1950', '1980')
seasons = ("Full_Season", 'MJJ', 'JJA', 'JAS', 'ASO', 'SON')

min_max_months = [[7,11],[5,7],[6,8],[7,9],[8,10],[9,11]]

variable = 'hgt200'
i = '1950'
s = seasons[0]

cases = ['DMI_sim_pos', 'DMI_sim_neg', 'DMI_neg', 'DMI_pos', 'DMI_un_pos', 'DMI_un_neg',
         'N34_pos', 'N34_neg', 'N34_un_pos', 'N34_un_neg']
c = cases[0]
for c in cases:
    for i in start:
        print(i)
        data = xr.open_dataset(data_dir + variable + '.nc')
        print('Open ' + variable + '.nc')
        if variable == 'hgt200':
            print('drop level')
            data = data.drop('level')
        # ------------------------------------------------------------------------------------------------------------------#
        if len(data.sel(lat=slice(-90, 20)).lat.values) == 0:
            data = data.sel(time=slice(i + '-01-01', '2020-12-01'), lat=slice(20, -90))
        else:
            data = data.sel(time=slice(i + '-01-01', '2020-12-01'), lat=slice(-90, 20))
        # ------------------------------------------------------------------------------------------------------------------#
        count = 0
        for s in seasons:

            mmonth = min_max_months[count]

            def PermuDatesComposite(n, data=data, mmonth=mmonth, season_name=s):
                mmin = mmonth[0]
                mmax = mmonth[-1]
                rn = np.random.RandomState(616)

                for a in n:
                    dates_rn = rn.permutation(neutro_concat)
                    neutro_new = dates_rn[0:len(neutro)]
                    data_new = dates_rn[len(neutro):]

                    neutro_comp = CompositeSimple(original_data=data, index=neutro_new,
                                                  mmin=mmin, mmax=mmax)
                    data_comp = CompositeSimple(original_data=data, index=data_new,
                                                mmin=mmin, mmax=mmax)

                    if a == n[0]:
                        comp = data_comp - neutro_comp
                        comp = comp.expand_dims(time=[a])
                        comp_concat = comp
                    else:
                        comp = data_comp - neutro_comp
                        comp = comp.expand_dims(time=[a])
                        comp_concat = xr.concat([comp_concat,comp],dim='time')

                    print(a)

                comp_concat.to_netcdf('/datos/luciano.andrian/ncfiles/nc_comps/' + 'Comps_' +
                               str(int(a)) + '.nc')
                del comp_concat


            aux = xr.open_dataset(nc_date_dir + 'Composite_' + i + '_2020_' + s + '.nc')
            neutro = aux.Neutral
            data_to_concat = aux[c]
            del aux

            M = NumberPerts(data_to_concat, neutro, 0)

            if data_to_concat[0] != 0:
                neutro_concat = np.concatenate([neutro, data_to_concat])
                from multiprocessing.pool import ThreadPool
                pool = ThreadPool(20)
                pool.map_async(PermuDatesComposite, [n for n in M])
                #[pool.apply_async(PermuDatesComposite, args=(n, data, mmonth)) for n in M]
                pool.close()
########################################################################################################################

            # aux = xr.open_mfdataset('/datos/luciano.andrian/ncfiles/nc_comps/*.nc', parallel=True
            #                         , chunks=100,
            #                         combine='nested', concat_dim="time", coords="different",
            #                         compat="broadcast_equals")
            #
            # qt = aux.persist().quantile([.05, .95], dim='time', interpolation='linear')
            #
            # qt.to_netcdf('/datos/luciano.andrian/ncfiles/nc_quantiles/' +
            #              variable + '_quantiles_'+ c + '_' + i + '-2020.nc')

########################################################################################################################
import xarray as xr
import dask
import numpy as np
from multiprocessing.pool import ThreadPool
with dask.config.set(schedular='threads', pool=ThreadPool(20)):
    aux = xr.open_mfdataset('/datos/luciano.andrian/ncfiles/nc_comps/Comps_*.nc', parallel=True
                            , chunks={'time':-1, 'lat':147,'lon':240},
                            combine='nested', concat_dim="time", coords="different",
                            compat="broadcast_equals")
    #aux = aux['var'].astype(np.float32)
    print('quantiles')
    aux = aux.chunk({'time':-1})
    qt = aux.quantile([.05,.95],dim='time', interpolation='linear')
    aux = np.quantile(aux['var'].persist(),[.05,.95], axis=0)
    #qt = aux.load().quantile([.05, .95], dim='time', interpolation='linear')
    # qt.to_netcdf('/datos/luciano.andrian/ncfiles/nc_quantiles/' +
    #              variable + '_quantiles_' + c + '_' + i + '-2020.nc')
