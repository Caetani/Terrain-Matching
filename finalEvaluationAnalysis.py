import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from generalParameters import files
from matplotlib import rcParams
from scipy.stats import hmean


def showDEMboxPlot(df, attribute, title, x_label, y_label, showBoxPlot=False):
    figSize = 5
    horizontalRatio = 1.2
    plotFonteSize = figSize*3
    smallFontRatio = 0.75
    titleFontRatio = 1.25
    fontName = 'Times New Roman'
    font = {'fontname':fontName}
    colors = ['#2066a8', '#8cc5e3', '#3594cc']
    rcParams.update({'font.size': smallFontRatio*plotFonteSize})
    plt.rcParams["font.family"] = fontName

    rs = df.loc[df['fileName'] == r'Datasets\RS\merged_map.tif']
    cy = df.loc[df['fileName'] == r'Datasets\Cyprus\Cyprus_data.tif']
    ok = df.loc[df['fileName'] == r'Datasets\USGS\OK_Panhandle.tif']

    fig, ax = plt.subplots(figsize=(horizontalRatio*figSize, 1.2*figSize))#, num=figureNumber)
    boxPlot = ax.boxplot([rs[attribute][:], cy[attribute][:], ok[attribute][:]], whis=1.5, patch_artist=True, notch=True, bootstrap=10_000)

    for patch, color in zip(boxPlot['boxes'], colors):
        patch.set_facecolor(color)
    
    for flier, color in  zip(boxPlot['fliers'], colors):
        flier.set(marker='o', color=color, markerfacecolor=color, markeredgecolor='black')

    for median in boxPlot['medians']:
        median.set_color('black') 

    ax.set_xticklabels(['Rio Grande\ndo Sul', 'Chipre', 'Oklahoma'], fontsize=smallFontRatio*plotFonteSize, **font)
    ax.set_title(title, fontsize=titleFontRatio*plotFonteSize, **font, weight="bold")
    ax.set_xlabel(x_label, fontsize=plotFonteSize, **font)
    ax.set_ylabel(y_label, fontsize=plotFonteSize, **font)
    plt.grid(axis='y')
    
    if showBoxPlot:
        plt.tight_layout()  # Ajusta o layout para evitar sobreposição
        plt.show()
    return


def showDEMboxPlotTransparent(df, attribute, title, x_label, y_label, showBoxPlot=False):
    figSize = 5
    horizontalRatio = 1.2
    plotFonteSize = figSize*3
    smallFontRatio = 0.75
    titleFontRatio = 1.25
    fontName = 'Times New Roman'
    font = {'fontname':fontName}
    colors = ['#2066a8', '#8cc5e3', '#3594cc']
    rcParams.update({'font.size': smallFontRatio*plotFonteSize})
    plt.rcParams["font.family"] = fontName

    rs = df.loc[df['fileName'] == r'Datasets\RS\merged_map.tif']
    cy = df.loc[df['fileName'] == r'Datasets\Cyprus\Cyprus_data.tif']
    ok = df.loc[df['fileName'] == r'Datasets\USGS\OK_Panhandle.tif']

    fig, ax = plt.subplots(figsize=(horizontalRatio*figSize, 1.2*figSize))#, num=figureNumber)
    boxPlot = ax.boxplot([rs[attribute][:], cy[attribute][:], ok[attribute][:]], whis=1.5, patch_artist=True, notch=False, bootstrap=10_000)

    for patch, color in zip(boxPlot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_fill(None)

    for flier, color in zip(boxPlot['fliers'], colors):
        #flier.set(marker='o', color=color, markerfacecolor=color, markeredgecolor='black')
        flier.set(marker='x', color='grey', markerfacecolor='grey', markeredgecolor='black')

    for median in boxPlot['medians']:
        median.set_color('black') 

    for i in range(len(rs)):
        y = rs[attribute].iloc[i]
        x = np.random.normal(loc=1, scale=0.04)
        plt.scatter(x, y, c='blue', alpha=0.4)
    for i in range(len(cy)):
        y = cy[attribute].iloc[i]
        x = np.random.normal(loc=2, scale=0.04)
        plt.scatter(x, y, c='green', alpha=0.4)
    for i in range(len(ok)):
        y = ok[attribute].iloc[i]
        x = np.random.normal(loc=3, scale=0.04)
        plt.scatter(x, y, c='darkorange', alpha=0.4)

    ax.set_xticklabels(['Rio Grande\ndo Sul', 'Chipre', 'Oklahoma'], fontsize=smallFontRatio*plotFonteSize, **font)
    ax.set_title(title, fontsize=titleFontRatio*plotFonteSize, **font, weight="bold")
    ax.set_xlabel(x_label, fontsize=plotFonteSize, **font)
    ax.set_ylabel(y_label, fontsize=plotFonteSize, **font)
    plt.grid(axis='y')
    
    if showBoxPlot:
        plt.tight_layout()  # Ajusta o layout para evitar sobreposição
        plt.show()
    return


def showDEMboxPlotSucessfulMatches(df, attribute, title, x_label, y_label, showBoxPlot=False):
    figSize = 5
    horizontalRatio = 1.2
    plotFonteSize = figSize*3
    smallFontRatio = 0.75
    titleFontRatio = 1.25
    fontName = 'Times New Roman'
    font = {'fontname':fontName}
    colors = ['#2066a8', '#8cc5e3', '#3594cc']
    rcParams.update({'font.size': smallFontRatio*plotFonteSize})
    plt.rcParams["font.family"] = fontName

    sucessful = df.loc[df['totalBestMatches'] > 0]
    non_sucessful = df.loc[df['totalBestMatches'] == 0]

    fig, ax = plt.subplots(figsize=(horizontalRatio*figSize, 1.2*figSize))#, num=figureNumber)
    boxPlot = ax.boxplot([sucessful[attribute][:], non_sucessful[attribute][:]], whis=1.5, patch_artist=True, notch=False, bootstrap=10_000)

    for patch, color in zip(boxPlot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_fill(None
                       )
    for flier, color in  zip(boxPlot['fliers'], colors):
        #flier.set(marker='o', color=color, markerfacecolor=color, markeredgecolor='black')
        flier.set(marker='x', color='grey', markerfacecolor='grey', markeredgecolor='black')

    for median in boxPlot['medians']:
        median.set_color('black') 

    for i in range(len(sucessful)):
        y = sucessful[attribute].iloc[i]
        x = np.random.normal(loc=1, scale=0.04)
        plt.scatter(x, y, c='green', alpha=0.4)
    for i in range(len(non_sucessful)):
        y = non_sucessful[attribute].iloc[i]
        x = np.random.normal(loc=2, scale=0.04)
        plt.scatter(x, y, c='red', alpha=0.4)

    ax.set_xticklabels(['Correspondências\nbem-sucedidas (10 melhores)', 'Sem correspondências\nbem-sucedidas (10 melhores)'], fontsize=smallFontRatio*plotFonteSize, **font)
    ax.set_title(title, fontsize=titleFontRatio*plotFonteSize, **font, weight="bold")
    ax.set_xlabel(x_label, fontsize=plotFonteSize, **font)
    ax.set_ylabel(y_label, fontsize=plotFonteSize, **font)
    plt.grid(axis='y')
    
    if showBoxPlot:
        plt.tight_layout()  # Ajusta o layout para evitar sobreposição
        plt.show()
    return


def plotHistogram(df, attribute, numBins, title='default', x_label='default', y_label='default',showBinsValues=False, showPlot=False):
    figSize = 5
    horizontalRatio = 1.2
    plotFonteSize = figSize*3
    smallFontRatio = 0.8
    titleFontRatio = 1.1
    fontName = 'Times New Roman'
    font = {'fontname':fontName}
    colors = ['#2066a8', '#8cc5e3', '#3594cc']
    rcParams.update({'font.size': smallFontRatio*plotFonteSize})
    plt.rcParams["font.family"] = fontName

    n, bins, patches = plt.hist(df[attribute][:], color='darkseagreen', edgecolor='white', bins=numBins, align='left', rwidth=0.9)
    if showBinsValues:
        bins = bins[:-1]
        for i, num in enumerate(bins):
            if n[i] != 0:
                plt.text(num, 1, round(num, 1), ha='center', color='black')
    plt.ylabel(y_label, fontsize=plotFonteSize, **font)
    plt.xlabel(x_label, fontsize=plotFonteSize, **font)
    plt.title(title, fontsize=titleFontRatio*plotFonteSize, **font, weight="bold")
    plt.show()
    return


def showHistPerDEM(df, attribute, numBins=10, acumulado=True, title='default', x_label='default', y_label='default'):
    figSize = 5
    horizontalRatio = 1.2
    plotFonteSize = figSize*3
    smallFontRatio = 0.8
    titleFontRatio = 1.1
    fontName = 'Times New Roman'
    font = {'fontname':fontName}
    colors = ['#2066a8', '#8cc5e3', '#3594cc']
    rcParams.update({'font.size': smallFontRatio*plotFonteSize})
    plt.rcParams["font.family"] = fontName

    b, bins, patches = plt.hist([df.loc[df['fileName'] == r'Datasets\RS\merged_map.tif', attribute],
                                df.loc[df['fileName'] == r'Datasets\Cyprus\Cyprus_data.tif', attribute],
                                df.loc[df['fileName'] == r'Datasets\USGS\OK_Panhandle.tif', attribute]],
                                stacked=acumulado,
                                label=['Rio Grande do Sul', 'Chipre', 'Oklahoma'],
                                edgecolor='white',
                                bins=numBins)
    plt.legend()
    plt.ylabel(y_label, fontsize=plotFonteSize, **font)
    plt.xlabel(x_label, fontsize=plotFonteSize, **font)
    plt.title(title, fontsize=titleFontRatio*plotFonteSize, **font, weight="bold")
    plt.show()
    plt.show()
    return


def barPlotEvaluationMetricPerDEM(df, attributes):
    figSize = 5
    horizontalRatio = 1.2
    plotFonteSize = figSize * 3
    smallFontRatio = 1
    titleFontRatio = 1.2
    fontName = 'Times New Roman'
    font = {'fontname': fontName}
    colors = ['#2066a8', '#8cc5e3', '#3594cc']
    rcParams.update({'font.size': smallFontRatio * plotFonteSize})
    plt.rcParams["font.family"] = fontName

    barWidth = 0.2  # Largura de cada barra
    group_gap = 0.6  # Espaçamento entre os grupos de barras
    intra_group_gap = 0.05  # Espaçamento entre as barras dentro de cada conjunto
    
    goodResults = []
    badResults = []

    for attr in attributes:
        goodResults.append(round(len((df.loc[df[attr] >= 1, attr]))))
        badResults.append(round(len(df)-goodResults[-1]))

    # The x position of bars
    r1 = np.arange(len(goodResults))
    r2 = [x + barWidth for x in r1]
    
    # Create blue bars
    plt.bar(r1, goodResults, width=barWidth, color='palegreen', edgecolor='black', capsize=7, label='Métrica maior ou igual a 1')
    
    # Create cyan bars
    plt.bar(r2, badResults, width=barWidth, color='lightcoral', edgecolor='black', capsize=7, label='Métrica menor que 1')

    # general layout
    plt.xticks([r + barWidth for r in range(len(goodResults))], ['Função objetivo', 'Correspondências\nbem-sucedidas', 'Correspondências bem-\nsucedidas (10 melhores)'], fontsize=plotFonteSize, **font)
    plt.ylabel('Frequência', fontsize=plotFonteSize, **font)
    plt.legend(loc='center right')
    plt.grid(axis='y')
    plt.title("Desempenho das métricas de\navaliação das 60 amostras", fontsize=titleFontRatio*plotFonteSize, **font, weight="bold")
    plt.show()


if __name__ == '__main__':
    df = pd.read_excel("Results\Final Evaluation\Final Results.xlsx")
    '''plotHistogram(df=df,
                  attribute='resizeScale',
                  numBins=8,
                  title='Distribuição dos fatores de\nredimensionamento das amostras',
                  x_label='Fator de redimensionamento',
                  y_label='Frequência',
                  showBinsValues=False,
                  showPlot=True)'''
    '''plotHistogram(df=df,
                  attribute='pitch',
                  numBins=10,
                  title='Distribuição dos ângulos de\narfagem das amostras',
                  x_label='Ângulo de afragem (graus)',
                  y_label='Frequência',
                  showBinsValues=True,
                  showPlot=True)'''
    '''plotHistogram(df=df,
                  attribute='roll',
                  numBins=10,
                  title='Distribuição dos ângulos de\nrolagem das amostras',
                  x_label='Ângulo de rolagem (graus)',
                  y_label='Frequência',
                  showBinsValues=True,
                  showPlot=True)'''
    '''plotHistogram(df=df,
                  attribute='rotAngle',
                  numBins=36,
                  title='Distribuição dos ângulos de\nguinada das amostras',
                  x_label='Ângulo de guinada (graus)',
                  y_label='Frequência',
                  showBinsValues=False,
                  showPlot=True)'''
    '''showHistPerDEM(df=df,
                   attribute='score',
                   numBins=10,
                   acumulado=True,
                   title='Distribuição dos valores da função objetivo\ndas amostras',
                   x_label='Valor da função objetivo',
                   y_label='Frequência')'''
    '''showDEMboxPlot(df=df,
                      attribute='score',
                      title='Distribuições dos valores da função objetivo\npara cada base de dados',
                      x_label='Base de Dados',
                      y_label='Valor da função objetivo',
                      showBoxPlot=True)'''
    '''showDEMboxPlot(df=df,
                   attribute='inliers',
                   title='Distribuições dos número de correspondências\n bem-sucedidas para cada base de dados',
                   x_label='Base de Dados',
                   y_label='Correspondências bem-sucedidas',
                   showBoxPlot=True)'''
    '''showDEMboxPlot(df=df,
                   attribute='totalBestMatches',
                   title='Distribuições de correspondências bem-sucedidas\ndentre as 10 melhores correspondências',
                   x_label='Base de Dados',
                   y_label='Correspondências bem-sucedidas entre as 10 melhores',
                   showBoxPlot=True)'''
    '''showDEMboxPlot(df=df,
                   attribute='sigmaREM',
                   title='Desvio padrão da sub-região em metros\n(antes das distorções)',
                   x_label='Base de Dados',
                   y_label='Desvio padrão da elevação (m)',
                   showBoxPlot=True)'''
    '''plotHistogram(df=df,
                  attribute='subSize',
                  numBins=10,
                  title='Percentual da região ocupada\npela sub-região',
                  x_label='Tamanho da sub-região (%)',
                  y_label='Frequência',
                  showBinsValues=True,
                  showPlot=True)'''
    '''showDEMboxPlotSucessfulMatches(df=df,
                   attribute='sigmaREM',
                   title='Distribuições do desvio padrão de elevação\nda sub-região separadas por desempenho',
                   x_label='Desempenho',
                   y_label='Desvio padrão da elevação (m)',
                   showBoxPlot=True)'''
    '''showDEMboxPlotTransparent(df=df,
                   attribute='score',
                   title='Distribuições dos valores da função objetivo\npara amostras em cada base de dados',
                   y_label='Valor da função objetivo',
                   x_label='Base de dados',
                   showBoxPlot=True)'''
    '''showDEMboxPlotTransparent(df=df,
                   attribute='inliers',
                   title='Distribuições das quantidades de correspondências\nbem-sucedidas para amostras em cada base de dados',
                   y_label='Quantidade de correspondências\nbem-sucedidas',
                   x_label='Base de dados',
                   showBoxPlot=True)'''
    '''showDEMboxPlotTransparent(df=df,
                   attribute='totalBestMatches',
                   title='Distribuições de correspondências bem-sucedidas\ndentre as 10 melhores correspondências',
                   y_label='Quantidade de correspondências bem-sucedidas\ndentre as 10 melhores',
                   x_label='Base de dados',
                   showBoxPlot=True)'''
    barPlotEvaluationMetricPerDEM(df,
                                  attributes=['score', 'inliers', 'totalBestMatches'])
    