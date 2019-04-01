# -*- coding: utf-8 -*-
"""
Created on Wed Aug 01 14:39:59 2018

@author: Aline



LEITURA DOS ARQUIVOS DO SIMCosta

"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import ntpath
from windrose import WindroseAxes

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

#A quick way to create new windrose axes...#,radialaxis='%'
def new_axes():
    fig = plt.figure(figsize=(13, 8), dpi=80, facecolor='w', edgecolor='w')
    rect = [0.1, 0.1, 0.8, 0.8]
    ax = WindroseAxes(fig, rect) #axisbg='w'
    fig.add_axes(ax)
    return ax

#...and adjust the legend box
def set_legend(ax):
    l = ax.legend(borderaxespad=-0.10,bbox_to_anchor=[-0.1, 0.5],loc='centerleft',title="m")
    plt.setp(l.get_texts(), fontsize=20)


files = glob.glob("G:\\Aline\\BG\\SIMCOSTA\\Dados\\*.csv")

dateparse = lambda x: pd.datetime.strptime(x, '%Y %m %d %I %M %S %p')

d = {}

for f in files:
    d[ntpath.basename(f)[0:-4]] = pd.read_csv(f, sep=',',
                 header = 0,
                 parse_dates={'datahora': ['year', 'month', 'day', 'hour',
                                           'min', 'sec', 'am_pm']},
                 date_parser=dateparse,
                 squeeze=True,
                 index_col=0)

d['RJ_2_met_completo']['Average_Pressure'].plot()
d['RJ_1_met_completo']['Average_Pressure'].plot(style='.')


for n in d.keys():
    if "met" in n:
        plt.figure(figsize=[14, 7], dpi=300)
        plt.subplot(3,2,1)
        d[n]['Average_Pressure'].plot(style='.')
        plt.xticks([])
        plt.xlabel('')
        plt.ylabel(u'Pressão Ar (mb)')
        plt.grid(which='both')
        plt.subplot(3,2,2)
        d[n]['Average_Air_Temperature'].plot(style='.')
        plt.xticks([])
        plt.xlabel('')
        plt.ylabel(u'Temp. Ar (C)')
        plt.grid(which='both')
        plt.subplot(3,2,3)
        d[n]['Instantaneous_Humidity'].plot(style='.')
        plt.xticks([])
        plt.xlabel('')
        plt.ylabel(u'Umidade Ar (%)')
        plt.grid(which='both')
        plt.subplot(3,2,4)
        d[n]['Average_Humidity'].plot(style='.')
        plt.xticks([])
        plt.xlabel('')
        plt.ylabel(u'Umidade Média Ar (%)')
        plt.grid(which='both')
        plt.subplot(3,2,5)
        d[n]['Average_Dew_Point'].plot(style='.')
        plt.ylabel(u'Ponto de Orvalho (C)')
        plt.grid(which='both')
        plt.subplot(3,2,6)
        d[n]['Solar_Radiation'].plot(style='.')
        plt.ylabel(u'Radiação Solar (W/m²)')
        plt.grid(which='both')
        plt.suptitle(n)
        plt.savefig("G:\\Aline\\BG\\SIMCOSTA\\Figuras\\first_look_meteo_"+
                     n[0:4] + ".png", bbox_inches='tight')
        plt.close()



        plt.figure(figsize=[14, 7], dpi=300)
        plt.subplot(3,1,1)
        d[n]['Average_wind_speed'].plot(style='.')
        plt.xticks([])
        plt.xlabel('')
        plt.ylabel(u'Wind Speed (m/s)')
        plt.grid(which='both')
        plt.subplot(3,1,2)
        d[n]['Average_wind_direction'].plot(style='.')
        plt.xticks([])
        plt.xlabel('')
        plt.ylabel(u'Wind Direction (degree)')
        plt.grid(which='both')
        plt.subplot(3,1,3)
        d[n]['Last_sampling_interval_gust_speed'].plot(style='.')
        plt.ylabel(u'Gust speed (m/s)')
        plt.grid(which='both')
        plt.suptitle(n)
        plt.savefig("G:\\Aline\\BG\\SIMCOSTA\\Figuras\\first_look_vento_"+
                     n[0:4] + ".png", bbox_inches='tight')
        plt.close()

    elif "ondas" in n:
        plt.figure(figsize=[14, 7], dpi=300)
        plt.subplot(3,1,1)
        d[n]['Hsig'].plot()
        plt.xticks([])
        plt.xlabel('')
        plt.ylabel('Hs (m)')
        plt.grid(which='both')
        plt.subplot(3,1,2)
        d[n]['Peak_Period'].plot()
        plt.xticks([])
        plt.xlabel('')
        plt.ylabel('Tp (s)')
        plt.grid()
        plt.subplot(3,1,3)
        d[n]['Mean_Wave_Direction'].plot()
        plt.ylabel('Direc')
        plt.ylim([0,360])
        plt.grid()
        plt.suptitle(n)
        plt.savefig("G:\\Aline\\BG\\SIMCOSTA\\Figuras\\first_look_ondas_"+
                     n[0:4] + ".png", bbox_inches='tight')
        plt.close()
        @todo()    
        # nao esta funcionando ainda!!!    
    elif "ocean" in n:
        plt.figure(figsize=[14,7], dpi=300)
        for i in np.arange(20):
            i = i+1
            veloc = d[n]['bin'+str(i)+'_mag']/1000
            direc = d[n]['bin'+str(i)+'_dir']*np.pi/180
            [X,Y] = pol2cart(veloc, direc)
            


"****************************************************************************"
"                                                                            "
"                       CONSISTENCIA                                         "
"                                                                            "
"****************************************************************************"
# Removendo registros apos 2016-10-07 da boia RJ01 que estavam inconsistentes
d['RJ_1_met_completo'] = d['RJ_1_met_completo'][d['RJ_1_met_completo'].index <
                                                  '2016-10-07']

d['RJ_1_ondas_completo']= d['RJ_1_ondas_completo'][d['RJ_1_ondas_completo']['Hsig']<6]

d['RJ_1_ondas_completo']= d['RJ_1_ondas_completo'][d['RJ_1_ondas_completo'
                                                     ].index > '2015-09-17 12:00']

plt.figure(figsize=[14, 7], dpi=300)
plt.subplot(2,1,1)
d['RJ_1_ondas_completo']['Hsig'].plot()
plt.ylabel('Altura Significativa (m)', fontsize=5)
plt.xticks([])
plt.xlabel(' ')
plt.yticks(fontsize=6)
plt.subplot(2,1,2)
d['RJ_1_ondas_completo']['Peak_Period'].plot()
plt.ylabel(u'Período de Pico (s)'), fontsize = 5)
plt.xticks(fontsize = 6)
plt.yticks(fontsize=6)
plt.suptitle('RJ-01', fontsize = 8)


d['RJ_1_ondas_completo']['Peak_Period'].hist(bins=20)
plt.xlabel(u'Período (s)')

d['RJ_1_ondas_completo']['Hsig'].hist(bins=20)



"Dados de onda a cada 30 min. Dimunuindo para 1 valor por hora"
RJ04 = d['RJ_4_ondas_completo'].resample('1H')
RJ04 = RJ04.max()

RJ04diario = d['RJ_4_ondas_completo'].resample('1D')
RJ04diario = RJ04diario.max()

RJ04diario.Hsig.max()
RJ04diario[RJ04diario.Hsig > 3].count()
RJ04diario[RJ04diario.Hsig > 2.5].count()
RJ04diario[RJ04diario.Hsig > 2.0].count()
RJ04diario[RJ04diario.Hsig > 1.5].count()
RJ04diario[RJ04diario.Hsig > 1.0].count()


RJ04diario[RJ04diario.Hsig > 2.5]['Mean_Wave_Direction'].min()
RJ04diario[RJ04diario.Hsig < 2.0]['Mean_Wave_Direction'].max()


RJ04diario.Hmax.max()
RJ04diario[RJ04diario.Hmax > 5].count()
RJ04diario[RJ04diario.Hmax > 4].count()
RJ04diario[RJ04diario.Hmax > 3].count()
RJ04diario[RJ04diario.Hmax > 2].count()


"Dados de onda a cada 30 min. Dimunuindo para 1 valor por hora"
RJ03 = d['RJ_3_ondas_completo'].resample('1H')
RJ03 = RJ03.max()

RJ03diario = d['RJ_3_ondas_completo'].resample('1D')
RJ03diario = RJ03diario.max()

RJ03diario.Hsig.max()
RJ03diario[RJ03diario.Hsig > 3].count()
RJ03diario[RJ03diario.Hsig > 2.5].count()
RJ03diario[RJ03diario.Hsig > 2.0].count()
RJ03diario[RJ03diario.Hsig > 1.5].count()
RJ03diario[RJ03diario.Hsig > 1.0].count()


RJ03diario[RJ03diario.Hsig > 2.5]['Mean_Wave_Direction'].min()
RJ03diario[RJ03diario.Hsig < 2.0]['Mean_Wave_Direction'].max()


RJ03diario.Hmax.max()
RJ03diario[RJ03diario.Hmax > 7].count()
RJ03diario[RJ03diario.Hmax > 6].count()
RJ03diario[RJ03diario.Hmax > 5].count()
RJ03diario[RJ03diario.Hmax > 4].count()
RJ03diario[RJ03diario.Hmax > 3].count()
RJ03diario[RJ03diario.Hmax > 2].count()

RJ03diario.Peak_Period.max()
RJ03diario.Peak_Period.mean()


RJ03diario.Peak_Period.hist()
plt.xlabel(u'Período (s)')
plt.ylabel('Ocorrencia')
plt.title('RJ-03 Distribuição TP')


RJ04diario.Peak_Period.hist()
plt.xlabel(u'Período (s)')
plt.ylabel('Ocorrencia')
plt.title('RJ-04 Distribuição TP')

RJ01diario.Peak_Period.hist()
plt.xlabel(u'Período (s)')
plt.ylabel('Ocorrencia')
plt.title('RJ-01 Distribuição TP')





RJ03diario.Hsig.hist()
plt.xlabel(u'Hs (m)')
plt.ylabel('Ocorrencia')
plt.title('RJ-03 Distribuição Hs')

RJ04diario.Hsig.hist()
plt.xlabel(u'Hs (m)')
plt.ylabel('Ocorrencia')
plt.title('RJ-04 Distribuição Hs')

RJ01diario.Hsig.hist()
plt.xlabel(u'Hs (m)')
plt.ylabel('Ocorrencia')
plt.title('RJ-01 Distribuição Hs')


RJ03diario.Mean_Wave_Direction.hist()
plt.xlabel(u'Direção (graus)')
plt.ylabel(u'Ocorrência')
plt.title('RJ-03 Distribuição Direção')
plt.xlim([0, 360])


RJ04diario.Mean_Wave_Direction.hist()
plt.xlabel(u'Direção (graus)')
plt.ylabel(u'Ocorrência')
plt.title('RJ-04 Distribuição Direção')
plt.xlim([0, 360])

RJ01diario.Mean_Wave_Direction.hist()
plt.xlabel(u'Direção (graus)')
plt.ylabel(u'Ocorrência')
plt.title('RJ-01 Distribuição Direção')
plt.xlim([0, 360])



" RJ01"
d['RJ_1_ondas_completo']= d['RJ_1_ondas_completo'][d['RJ_1_ondas_completo']['Hsig']<6]

d['RJ_1_ondas_completo']= d['RJ_1_ondas_completo'][d['RJ_1_ondas_completo'
                                                     ].index > '2015-09-17 12:00']

RJ01 = d['RJ_1_ondas_completo'].resample('1H')
RJ01 = RJ01.max()

RJ01diario = d['RJ_1_ondas_completo'].resample('1D')
RJ01diario = RJ01diario.max()

RJ01diario.Hsig.max()
RJ01diario[RJ01diario.Hsig > 4].count()
RJ01diario[RJ01diario.Hsig > 3.5].count()
RJ01diario[RJ01diario.Hsig > 3].count()
RJ01diario[RJ01diario.Hsig > 2.5].count()
RJ01diario[RJ01diario.Hsig > 2].count()
RJ01diario[RJ01diario.Hsig > 1.5].count()
RJ01diario[RJ01diario.Hsig > 1].count()


RJ01diario[RJ01diario.Hsig > 2.5]['Mean_Wave_Direction'].min()
RJ01diario[RJ01diario.Hsig < 2.0]['Mean_Wave_Direction'].median()


RJ01diario.Hmax.max()
RJ01diario[RJ01diario.Hmax > 8].count()
RJ01diario[RJ01diario.Hmax > 7].count()
RJ01diario[RJ01diario.Hmax > 6.5].count()
RJ01diario[RJ01diario.Hmax > 6].count()
RJ01diario[RJ01diario.Hmax > 5].count()
RJ01diario[RJ01diario.Hmax > 4].count()
RJ01diario[RJ01diario.Hmax > 3].count()
RJ01diario[RJ01diario.Hmax > 2].count()


RJ01diario[RJ01diario.Hmax > 4]['Mean_Wave_Direction'].median()
RJ01diario[RJ01diario.Hsig < 2.0]['Mean_Wave_Direction'].median()

"Rosa de onda com o período"
ws = RJ02.Peak_Period
wd = RJ02.Mean_Wave_Direction

label_size = 14
plt.rcParams['xtick.labelsize'] = label_size
plt.rcParams['ytick.labelsize'] = label_size
ax = new_axes()
ax.bar(wd, ws, opening=0.8, edgecolor='black',
       bins=np.arange(ws.min(),ws.max(),3))
set_legend(ax)

#tabela de frequencias a partir da rosa
table = ax._info['table']
# contagem de ocorrencia por classe de periodo
Tp_dir_freq = np.sum(table, axis=1)
Tp_dir_freq = np.c_[np.arange(ws.min(),ws.max(),3), Tp_dir_freq]

"Rosa de onda com o Hs"
ws = RJ01.Hsig
wd = RJ01.Mean_Wave_Direction

label_size = 14
plt.rcParams['xtick.labelsize'] = label_size
plt.rcParams['ytick.labelsize'] = label_size
ax = new_axes()
ax.bar(wd, ws, opening=0.8, edgecolor='black',
       bins=np.arange(ws.min(),ws.max(),0.8))
set_legend(ax)

#tabela de frequencias a partir da rosa
table = ax._info['table']
# contagem de ocorrencia por classe de periodo
Hs_dir_freq = np.sum(table, axis=0)
Hs_freq = np.sum(table, axis=1)
Hs_freq = np.c_[np.arange(ws.min(),ws.max(),0.8), Hs_freq]



" selecionando o periodo de 15-9 a 17-09 "

met_crop= d['RJ_1_met_completo'][d['RJ_1_met_completo'].index > '2016-09-15 00:00']
met_crop= met_crop[met_crop.index < '2016-09-17 23:00']

onda_crop= d['RJ_1_ondas_completo'][d['RJ_1_ondas_completo'].index > '2016-09-15 00:00']
onda_crop= onda_crop[onda_crop.index < '2016-09-17 23:00']

plt.subplot(2,1,1)
met_crop.Average_wind_speed.plot()
onda_crop.Hsig.plot()
plt.legend()
plt.subplot(2,1,2)
met_crop.Average_wind_direction.plot(style='o')
onda_crop.Mean_Wave_Direction.plot(style='^')
plt.legend()
