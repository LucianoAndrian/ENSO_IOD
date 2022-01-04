import xarray as xr
import numpy as np
from multiprocessing.pool import ThreadPool
import os
import glob
import math
from datetime import datetime
import dask
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

start = ('1920', '1950')
seasons = ("Full_Season", 'JJA', 'ASO', 'SON')

min_max_months = [[7,11], [6,8],[8,10],[9,11]]

variables = ['hgt200', 'div', 'psl', 'sf', 'vp', 't_cru', 't_BEIC', 'pp_gpcc']

cases = ['DMI_sim_pos', 'DMI_sim_neg', 'DMI_neg', 'DMI_pos', 'DMI_un_pos', 'DMI_un_neg',
         'N34_pos', 'N34_neg', 'N34_un_pos', 'N34_un_neg']

#----------------------------------------------------------------------------------------------------------------------#

for v in variables:
    print('Variable: ' + v)

    for i in start:
        print('Período: ' + i + '- 2020')
        print('Open ' + v + '.nc')

        data = xr.open_dataset(data_dir + v + '1x1.nc')

        if v == 'hgt200':
            print('drop level')
            data = data.drop('level')
        elif v == 'psl':
            print('to hPa')
            data = data.__mul__(1 / 100)

        for c in cases:
            print(c)
            count = 0
            for s in seasons:
                print(s)

                files = glob.glob('/datos/luciano.andrian/ncfiles/nc_comps/*.nc')
                if len(files) != 0:
                    for f in files:
                        try:
                            os.remove(f)
                        except:
                            print('Error: ' + f)

                mmonth = min_max_months[count]

                def PermuDatesComposite(n, data=data, mmonth=mmonth):
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
                            comp_concat = xr.concat([comp_concat, comp], dim='time')

                    comp_concat.to_netcdf('/datos/luciano.andrian/ncfiles/nc_comps/' + 'Comps_' +
                                          str(int(a)) + '.nc')
                    del comp
                    del data_comp
                    del neutro_new
                    del comp_concat

                aux = xr.open_dataset(nc_date_dir + 'Composite_' + i + '_2020_' + s + '.nc')
                neutro = aux.Neutral
                data_to_concat = aux[c]
                aux.close()

                M = NumberPerts(data_to_concat, neutro, 0)

                if (data_to_concat[0] != 0):
                    neutro_concat = np.concatenate([neutro, data_to_concat])

                    hour = datetime.now().hour
                    if (hour > 20) | (hour < 7):
                        n_thread = 25
                        pool = ThreadPool(25)
                    else:
                        n_thread = 10
                        pool = ThreadPool(10)

                    print('Threads: ', n_thread)

                    pool.map(PermuDatesComposite, [n for n in M])
                    pool.close()

                    aux = xr.open_mfdataset('/datos/luciano.andrian/ncfiles/nc_comps/Comps_*.nc', parallel=True
                                            , #chunks={'time': -1},  # 'lat':147,'lon':240},
                                            combine='nested', concat_dim="time", coords="different",
                                            compat="broadcast_equals")
                    # aux = aux['var'].astype(np.float32)
                    print('quantiles')
                    aux = aux.chunk({'time': -1})
                    qt = aux.quantile([.05, .95], dim='time', interpolation='linear')
                    qt.to_netcdf('/datos/luciano.andrian/ncfiles/nc_quantiles/' + v + '_' + c + '_' + i +
                                 '_2020' + '_' + s + '.nc', compute=True)
                    aux.close()
                    del qt

                    # with dask.config.set(schedular='threads', pool=ThreadPool(10)):

                else:
                    print('no ' + c)

            count += 1





########################################################################################################################
