# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 12:16:00 2018

@author: Aline

Criando figuras para analise das ondas
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import ntpath
from windrose import WindroseAxes
import seaborn as sns
import scipy.stats as stt

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


fonte = 14


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
d['RJ_1_ondas_completo']= d['RJ_1_ondas_completo'][d['RJ_1_ondas_completo'
                                                     ]['Hsig'] > 0]


"Rosa de onda com o período"

for n in d.keys():
    if "ondas" in n:
        ws = d[n]['Hsig']
        wd = d[n]['Mean_Wave_Direction']

        label_size = 14
        plt.rcParams['xtick.labelsize'] = label_size
        plt.rcParams['ytick.labelsize'] = label_size
        ax = new_axes()
        ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='black',
               bins=[0,0.8,1.6,2.4,3.2,4.0])
        set_legend(ax)
        plt.title(n[0:4])
        plt.savefig("G:\\Aline\\BG\\SIMCOSTA\\Figuras\\rosa_ondas_"+
                     n[0:4] + ".png", bbox_inches='tight', transparent=True)


" Fazendo distribuicao conjunta"

for n in d.keys():
    if "ondas" in n:
        df0 = d[n]
        # agrupando os dados pela altura significativa e o periodo em classes
        # pre determinadas usando o pd.cut
        # faz a contagem dos registros presente em casa uma das classes
        b = df0.groupby([pd.cut(df0.Hsig, np.arange(0, 5, 0.5)),
               pd.cut(df0.Peak_Period, np.arange(0,30,3))]).count()
        
        # passando todo mundo para a referencia de 1 ano
        # 1 ano teria 17520 registros com frequencia amostral de 30 min
        fator = len(df0)/17520
        
        df = pd.DataFrame()
        df['Hs']=b.index.get_level_values(level=0)
        df['Tp']=b.index.get_level_values(level=1)
        
        # fator transforma a serie para o periodo de 1 ano. 
        # 48 eh o numero de registro em um dia
        # df['count] tem unidade dias??
        df['count'] = ((b.MessageID.values)/fator)/48
        df['porcentagem'] = (df['count'] * 100) /365
        
        pivot = df.pivot(index='Hs', columns= 'Tp', values='count')
        
        labels = df.pivot(index='Hs', columns= 'Tp', values='porcentagem')
        cmap = sns.cm.rocket_r
        f, ax = plt.subplots(figsize=(13.6,6.38))
        sns.heatmap(pivot, 
                    annot=labels, 
                    fmt='.2f', 
                    cmap='YlGnBu', 
                    vmin=0, 
                    vmax=120,
                    linecolor='k',
                    linewidths=.5)
        
        # configurando a figura
        for t in ax.texts: t.set_text(t.get_text() + ' %')
        plt.gca().invert_yaxis()
        plt.title(n[0:4])
        plt.ylabel(u'Altura Significativa (m)')
        plt.xlabel(u'Período de Pico (s)')
        plt.savefig("G:\\Aline\\BG\\SIMCOSTA\\Figuras\\dist_conjunta"+
                     n[0:4] + ".png", bbox_inches='tight', dpi = 300)
        plt.close()


" fazendo distribuicao conjunta a partir do evento maximo do dia"

for n in d.keys():
    if "ondas" in n:
        df = d[n]
        # agrupando os dados pela altura significativa e o periodo em classes
        # pre determinadas usando o pd.cut
        # faz a contagem dos registros presente em casa uma das classes
        df = df.resample('1D').max()
        b = df.groupby([pd.cut(df.Hsig, np.arange(0, 5, 0.5)),
               pd.cut(df.Peak_Period, np.arange(0,30,3))]).count()
        
        # passando todo mundo para a referencia de 1 ano
        # 1 ano teria 17520 registros com frequencia amostral de 30 min
        fator = len(df)/365
        
        df = pd.DataFrame()
        df['Hs']=b.index.get_level_values(level=0)
        df['Tp']=b.index.get_level_values(level=1)
        df['count'] = (b.MessageID.values/fator)
        df['porcentagem'] = df['count'] * (100 /365)
        
        pivot = df.pivot(index='Hs', columns= 'Tp', values='count')
        
        labels = df.pivot(index='Hs', columns= 'Tp', values='porcentagem')
        cmap = sns.cm.rocket_r
        f, ax = plt.subplots(figsize=(13.6,6.38))
        sns.heatmap(pivot, 
                    annot=labels, 
                    fmt='.2f', 
                    cmap='YlGnBu', 
                    vmin=0, 
                    vmax=120,
                    linecolor='k',
                    linewidths=.5)
        
        # configurando a figura
        for t in ax.texts: t.set_text(t.get_text() + ' %')
        plt.gca().invert_yaxis()
        plt.title(n[0:4])
        plt.ylabel(u'Altura Significativa (m)')
        plt.xlabel(u'Período de Pico (s)')
        plt.savefig("G:\\Aline\\BG\\SIMCOSTA\\Figuras\\dist_conjunta_resample_dia"+
                     n[0:4] + ".png", bbox_inches='tight', dpi = 300)
        plt.close()



" fazendo CDF  WORKING"
plt.figure(figsize=(13.6,6.38))
for n in d.keys():
    if "ondas" in n:
        df = d[n]       
        df.Hsig.hist(normed=True, 
                     cumulative=True, 
                     histtype=u'step', 
                     label= n[0:4],
                     linewidth=2)
        plt.title('CDF Hs')
        plt.xlabel('Altura (m)', 
                   fontsize = fonte)
        plt.ylabel(u'Frequência Acumulada de Ocorrência', 
                   fontsize = fonte)
        plt.legend()
        plt.savefig("G:\\Aline\\BG\\SIMCOSTA\\Figuras\\CDF_Hs.png", 
                    bbox_inches='tight', 
                    dpi = 300)
        plt.close()

plt.figure(figsize=(13.6,6.38))
for n in d.keys():
    if "ondas" in n:
        df = d[n]
        df.Peak_Period.hist(normed=True, 
                            cumulative=True, 
                            histtype=u'step', 
                            label= n[0:4],
                            linewidth=2)
        plt.title('CDF Tp')
        plt.xlabel(u'Período de Pico (s)', 
                   fontsize = fonte)
        plt.ylabel(u'Frequência Acumulada de Ocorrência', 
                   fontsize = fonte)
        plt.legend()
        plt.savefig("G:\\Aline\\BG\\SIMCOSTA\\Figuras\\CDF_Tp.png", 
                    bbox_inches='tight', 
                    dpi = 300)
        plt.close()


" Fazendo figuras de temperatura"
for n in d.keys():
    if "4_ocean" in n:
        plt.figure(figsize=(13.6,6.38))
        df = d[n]
        df.t2.plot(style='.')
        plt.title(u'Temperatura da água ' + n[0:4])
        plt.xlabel(u'', fontsize = fonte)
        plt.ylabel(u'Temperatura ($\degree$ C)')
        plt.legend()
        plt.grid()
        plt.savefig("G:\\Aline\\BG\\SIMCOSTA\\Figuras\\temperatura_agua"+
                     n[0:4] + ".png", bbox_inches='tight', dpi = 300)
        plt.close()
    elif "ocean" in n:
        plt.figure(figsize=(13.6,6.38))
        df = d[n]
        df.t1.plot(style='o')
        df.t2.plot(style='.')
        plt.title(u'Temperatura da água ' + n[0:4])
        plt.xlabel(u'', fontsize = fonte)
        plt.ylabel(u'Temperatura ($\degree$ C)')
        plt.legend()
        plt.grid()
        plt.savefig("G:\\Aline\\BG\\SIMCOSTA\\Figuras\\temperatura_agua"+
                     n[0:4] + ".png", bbox_inches='tight', dpi = 300)
        plt.close()


"Fazendo figura de extenção das campanhas"
plt.figure(figsize=(13.6,6.38))
for n in d.keys():
    if "ocean" in n:
        fator = 1 + ((int(n[3]))/10)
        plt.plot(d[n].index, 
                 fator *np.ones(len(d[n])), 
                 label=n[0:4], 
                 linewidth=10)
        plt.legend(fontsize=fonte, ncol=1)
        plt.grid(axis='x', linestyle='dashed')
        plt.tick_params(axis='y', left=False, labelleft=False)
        plt.tick_params(axis='x', labelsize=fonte)
plt.savefig("G:\\Aline\\BG\\SIMCOSTA\\Figuras\\periodo_medicao_boias.png", 
            bbox_inches='tight', 
            dpi = 300)
plt.close()


" Separando magnitude de direçao em duas variaveis diferentes "
figuras = False
for n in d.keys():
    if "ocean" in n:
        mag = pd.DataFrame()
        direc = pd.DataFrame()
        b = d[n]['n_of_bins'][0]
        bins = np.arange(1, b+1)
        for j in bins:
            mag['bin'+str(int(j))] = d[n]['bin'+str(int(j))+'_mag']
            direc['bin'+str(int(j))] = d[n]['bin'+str(int(j))+'_dir']
        #plt.figure()
        if figuras == True:
            mag.plot(style='.')
            plt.legend(' ')
            plt.title(n[0:4])
            plt.savefig("G:\\Aline\\BG\\SIMCOSTA\\Figuras\\corrente_" + n[0:4] + 
                        "_bruto.png", 
                        bbox_inches='tight', 
                        dpi = 300)
            plt.close('all')
        
        " consistencia "
        # pegando os indices que sao maiores que um determinado valor
        indices = mag<2000
        mag = mag[indices]
        direc = direc[indices]
        #plt.figure()
        if figuras == True:
            mag.plot(style='.')
            plt.title(n[0:4] + ' serie filtrada')
            plt.legend(' ')
            plt.savefig("G:\\Aline\\BG\\SIMCOSTA\\Figuras\\corrente_" + n[0:4] + 
                "_filtrada.png", 
                bbox_inches='tight', 
                dpi = 300)
            plt.close('all')
        
        # fazendo a media na vertical
        mag_media_vertical = mag.mean(axis=1)
        direc_media_vertical = direc.mean(axis=1)
        # transformando em m/s
        mag_media_vertical = mag_media_vertical/1000
        #plt.figure()
        if figuras == True:
            mag_media_vertical.plot(style='.', c='steelblue')
            plt.ylabel('Corrente (m/s)')
            plt.title(u'Corrente - média vertical')
            plt.savefig("G:\\Aline\\BG\\SIMCOSTA\\Figuras\\corrente_" + n[0:4] + 
                "_media_vertical.png", 
                bbox_inches='tight', 
                dpi = 300)
            plt.close('all')
        
        "*****************************************************"
        "  Figuras:                                           "
        "          - Rosa de correntes: media vertical        "
        " Rosa de correntes - fazer igual minha dissertacao???"
        "*****************************************************"
        if figuras == True:
            ws = mag_media_vertical.values
            wd = direc_media_vertical.values
    
            label_size = 14
            plt.rcParams['xtick.labelsize'] = label_size
            plt.rcParams['ytick.labelsize'] = label_size
            ax = new_axes()
            ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='black',
                   bins=[0,0.1, 0.2, 0.3, 0.5, 1, 2])
            set_legend(ax)
            plt.title(n[0:4])
            plt.savefig("G:\\Aline\\BG\\SIMCOSTA\\Figuras\\rosa_correntes_"+
                         n[0:4] + ".png", bbox_inches='tight', transparent=True)
            
        "*****************************************************"
        "  Figuras:                                           "
        "          - Rosa de correntes: medicoes em superficie"
        "                                                     "
        "*****************************************************"
        ws = (mag['bin1'].values)/1000
        wd = direc['bin1'].values

        label_size = 14
        plt.rcParams['xtick.labelsize'] = label_size
        plt.rcParams['ytick.labelsize'] = label_size
        ax = new_axes()
        ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='black',
               bins=[0,0.1, 0.2, 0.3, 0.5, 1, 2])
        set_legend(ax)
        plt.title(n[0:4] + u' - Superfície (bin1)')
        plt.savefig("G:\\Aline\\BG\\SIMCOSTA\\Figuras\\rosa_correntes_sup"+
                     n[0:4] + ".png", bbox_inches='tight', transparent=True)
        
        " MEIO"
        ws = (mag['bin10'].values)/1000
        wd = direc['bin10'].values

        label_size = 14
        plt.rcParams['xtick.labelsize'] = label_size
        plt.rcParams['ytick.labelsize'] = label_size
        ax = new_axes()
        ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='black',
               bins=[0,0.1, 0.2, 0.3, 0.5, 1, 2])
        set_legend(ax)
        plt.title(n[0:4] + u' - Camada Intermediária (bin10)')
        plt.savefig("G:\\Aline\\BG\\SIMCOSTA\\Figuras\\rosa_correntes_meio"+
                     n[0:4] + ".png", bbox_inches='tight', transparent=True)
        
        " FUNDO"
        ws = (mag['bin16'].values)/1000
        wd = direc['bin16'].values

        label_size = 14
        plt.rcParams['xtick.labelsize'] = label_size
        plt.rcParams['ytick.labelsize'] = label_size
        ax = new_axes()
        ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='black',
               bins=[0,0.1, 0.2, 0.3, 0.5, 1, 2])
        set_legend(ax)
        plt.title(n[0:4] + u' - Fundo (bin16)')
        plt.savefig("G:\\Aline\\BG\\SIMCOSTA\\Figuras\\rosa_correntes_meio"+
                     n[0:4] + ".png", bbox_inches='tight', transparent=True)       
        
#        "Rosa de correntes iguais ao do meu mestrado"
#        ws = mag_media_vertical.resample('3H').mean().values
#        wd = direc_media_vertical.resample('3H').mean().values
#        
#        ax = plt.subplot(111, projection='polar')
#        ax.set_theta_zero_location("N")
#        ax.set_theta_direction(-1)
#        [ax.arrow(0,0,wd[i], ws[i], length_includes_head=True, fc='k', ec='k') 
#        for i in range(len(ws))]
#        plt.savefig(pathname + '\\corrente_seta_preta' + n[0:4] + 
#                    '.png',  transparent=True)
#        #mag = mag.resample('3H')
#        #mag = mag.mean()
        
        
        

"          - Ver se tem diferença sup. vs fundo que   "
"                valha a pena mostrar                 "
        
        
        
        
        
        
        
"          - Serie temporal RJ02 e RJ03 no periodo que"
"                se sobrepoe (ver defasagem da mare na"
"                camada intermediaria)                "
plt.figure(figsize=(13.6,6.38))
for n in d.keys():
    if "2_ocean" in n or "3_ocean" in n:
        mag = pd.DataFrame()
        direc = pd.DataFrame()
        b = d[n]['n_of_bins'][0]
        bins = np.arange(1, b+1)
        for j in bins:
            mag['bin'+str(j)] = d[n]['bin'+str(int(j))+'_mag']
            direc['bin'+str(j)] = d[n]['bin'+str(int(j))+'_dir']
        mag_crop = mag[mag.index > '2016-09-21']
        mag_crop = mag_crop[mag_crop.index < '2016-10-21']
        mag_crop['bin1'].plot(label=n[0:4])
        plt.hold(True)
        
#        direc_crop = direc[direc.index > '2016-09-21']
#        direc_crop = direc_crop[direc_crop.index < '2016-10-21']
#        direc_crop['bin10'].plot(label=n[0:4])
#        plt.hold(True)


"          - Fazer analise do tempo que a onda passa  "
"                com valor superior a X               "
"*****************************************************"

        
        
        
        

""" 
Avaliando periodo com intensidicação de correntes. Na boia RJ04, 
no periodo de 18 de janeiro de 2018 a 31 de janeiro de 2018 foram registradas 
correntes de até 1.2 m/s com períodos de intensificação de 10 a 12hrs.
Fazer uma análise do espectro de frequencias.
"""

mag_crop = mag[mag.index > '2018-01-18']

mag_crop = mag_crop[mag_crop.index < '2018-01-31']

direc_crop = direc[direc.index < '2018-01-31']

direc_crop = direc_crop[direc_crop.index > '2018-01-18']

ws = (mag_crop['bin1'].values)/1000
wd = direc_crop['bin1'].values

label_size = 14
plt.rcParams['xtick.labelsize'] = label_size
plt.rcParams['ytick.labelsize'] = label_size
ax = new_axes()
ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='black',
       bins=[0,0.1, 0.2, 0.3, 0.5, 1, 2])
set_legend(ax)
plt.title(n[0:4] + u' - Superfície (bin1)')













