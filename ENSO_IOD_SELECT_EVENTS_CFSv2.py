"""
A partir de LoadLeads_CFSv2.py y ENSO_IOD_DMI_N34_CFSv2.py
Selecciona para cada miembro de ensambe y lead los campos en los que se dan los eventos
"""
########################################################################################################################
import xarray as xr
########################################################################################################################
dates_dir = '/datos/luciano.andrian/ncfiles/NMME_CFSv2/index_r/z'
dir_leads = '/datos/luciano.andrian/ncfiles/NMME_CFSv2/'
out_dir = '/pikachu/datos/luciano.andrian/cases/'
########################################################################################################################
mmonth_seasons = [0,1,2,3] #son las de abajo...
seasons = ['JJA', 'JAS', 'ASO', 'SON']
variables = ['prec', 'tref']
cases = ['DMI_sim_pos', 'DMI_sim_neg', 'DMI_neg', 'DMI_pos', 'DMI_un_pos', 'DMI_un_neg',
         'N34_pos', 'N34_neg', 'N34_un_pos', 'N34_un_neg', 'Neutral']

# Fechas para cada case en cases (lo de arriba)
#[r,ms,l,y]
dates = xr.open_dataset(dates_dir + 'dates_dmi_n34_cfsv2.nc')

#conjuntos de leads
conjuntos = ([0],[0,1],[0,1,2,3],[4,5,6,7],[0,1,2,3,4,5,6,7])
conj_name= ('0','1','3','7', 'todo')

for v in variables:
    print(v)

    # pp en mm/day
    if v == 'prec':
        factor=30
    else:
        factor=1
    for c in cases: #loop en los eventos
        case = dates[c]

        for ms in mmonth_seasons: #loop en las seasons
            season_name = seasons[ms]
            print(season_name)
            cj_count=0
            for cj in conjuntos:# loop en los conjuntos de leads
                print(cj)
                count = 0
                for l in cj: #loop en los leads de cada conjunto
                    print(l)
                    #abre salida de LoadLeads.py
                    data_r = xr.open_dataset(dir_leads + v + '_CFSv2_' + season_name + '_Lead_' + str(l) + '.nc').__mul__(factor)

                    for r in range(0, 24): #loop en los miembros de ensamble
                        # de cada "case" selecciona el miembro de ensamble, la season(ms) y lead
                        case_date = case[r, ms, l, :]
                        # los nan se usaron solo para crear el xr.Dataset y que no joda con la longitud de los años
                        case_date = case_date.dropna('years').values

                        # selecciona en data_r los miembros de ensamble y anios en case, case_date


                        if (ms == 0 or ms == 1) and l>5 and len(case_date) != 0 and case_date[0]==1982:
                            data_r_case = data_r.sel(r=r + 1, Year=case_date[1::])
                        else:
                            data_r_case = data_r.sel(r=r + 1, Year=case_date)

                        #
                        # if c == 'Neutral':
                        #     if ms == 0 and l > 5:
                        #         print('Neutral Events')
                        #         if len(case_date) !=0 and case_date[0]==1982:
                        #             # Rec. Los neutro se calculan mirando en que años
                        #             # (dentro de periodo) no hay DMI o N34. Para los
                        #             # pronos de JJA con lead 7, nunca hay eventos 1982.
                        #             # En esos casos tanto el indice como los leads agrupados
                        #             # pasan a mirar diciembre/noviembre de 1982 para los pronos de 1983
                        #             # y son guardados con ese año 1983
                        #             #
                        #             # corto, no existen lead =7 para JJA en 1982
                        #             data_r_case = data_r.sel(r=r + 1, Year=case_date[1::])
                        # else:
                        #     data_r_case = data_r.sel(r=r + 1, Year=case_date)

                        #guardado
                        if len(data_r_case.Year) != 0:

                            if count == 0:
                                data_f = xr.Dataset(
                                    data_vars=dict(
                                        var=(['case', 'lat', 'lon'], data_r_case[v].values)))
                                count += 1
                            else:
                                data = xr.Dataset(
                                    data_vars=dict(
                                        var=(['case', 'lat', 'lon'], data_r_case[v].values)))
                                data_f = xr.concat([data_f, data], dim='case')

                            data_f.to_netcdf(
                                out_dir + v + '_' + season_name + '_' + c + '_' + conj_name[cj_count] + '.nc')
                            del data_f

                        else:
                            print('NO DATA: ' + v + ' in ' + c + ' at: ' + '\n' + 'cj: ' + str(cj) +
                                  ' Lead: ' + str(l) + ' r: ' + str(r)  +  ' ' + season_name)
                cj_count += 1