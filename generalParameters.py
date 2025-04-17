'''
    Script with commmom definitions of terrain matching algorithm for optimization.
'''

import numpy as np
import pandas as pd
import cv2
from commom import *

files = [r'Datasets\RS\merged_map.tif',
         r'Datasets\Cyprus\Cyprus_data.tif',
         r'Datasets\USGS\OK_Panhandle.tif']

datasetsResolutions = [30, 5, 1]
datasetsNoises      = [15, 2.5, 0.5]

SPACE_sigmaDEM                    = np.arange(1, 7+1, 1)
SPACE_sigmaREM                    = np.arange(1, 7+1, 1)
SPACE_windowSizeScaleDEM          = np.arange(1, 10+1, 1)
SPACE_windowSizeScaleREM          = np.arange(1, 10+1, 1)
SPACE_patchSizeDEM                = np.arange(5, 55+1, 10)
SPACE_patchSizeREM                = np.arange(5, 55+1, 10)
SPACE_nLevelsDEM                  = np.arange(4, 12+1, 1)
SPACE_nLevelsREM                  = np.arange(4, 12+1, 1)
SPACE_initLevelDEM                = np.arange(0, 3+1, 1)
SPACE_initLevelREM                = np.arange(0, 3+1, 1)
SPACE_scaleFactorDEM              = np.arange(1.1, 1.6+0.1, 0.1)
SPACE_scaleFactorREM              = np.arange(1.1, 1.6+0.1, 0.1)
SPACE_fastThresholdDEM            = np.arange(5, 45+1, 5)
SPACE_fastThresholdREM            = np.arange(5, 45+1, 5)
SPACE_crossCheck                  = np.arange(0, 1+1, 1)

SPACE_ALL = [SPACE_sigmaDEM,
             SPACE_sigmaREM,
             SPACE_windowSizeScaleDEM,
             SPACE_windowSizeScaleREM,
             SPACE_patchSizeDEM,
             SPACE_patchSizeREM,
             SPACE_nLevelsDEM,
             SPACE_nLevelsREM,
             SPACE_initLevelDEM,
             SPACE_initLevelREM,
             SPACE_scaleFactorDEM,
             SPACE_scaleFactorREM, 
             SPACE_fastThresholdDEM,
             SPACE_fastThresholdREM,
             SPACE_crossCheck]

GA_num_parents_mating             = np.arange(0.4, 0.8+0.2, 0.2)
GA_parent_selection_type          = np.arange(0, 5+1, 1)
GA_keep_elitism                   = np.arange(0, 2+1, 1)
GA_crossover_type                 = np.arange(0, 3+1, 1)
GA_crossover_probability          = np.arange(0.4, 1+0.3, 0.3)
GA_mutation_type                  = np.arange(0, 0+1, 1)
GA_mutation_num_genes_init        = np.arange(0.1, 0.5+0.1, 0.2)
GA_mutation_num_genes_end         = np.arange(0.2, 1+0.1, 0.4)

GA_PARAMETERS_SPACES = [GA_num_parents_mating,
                        GA_parent_selection_type,
                        GA_keep_elitism,
                        GA_crossover_type,
                        GA_crossover_probability,
                        GA_mutation_num_genes_init,
                        GA_mutation_num_genes_end]

GA_PARAMETERS_STRINGS = [
    'num_parents_mating',
    'parent_selection_type',
    'keep_elitism',
    'crossover_type',
    'crossover_probability',
    'mutation_num_genes_init',
    'mutation_num_genes_end'
]

def GA_getCrossoverType(n):
    crossoverTypeDict = {
        0: "single_point",
        1: "two_points",
        2: "uniform",
        3: "scattered",
    }
    return crossoverTypeDict[n]

def GA_getParentSelectionType(n):
    parentSelectionDict = {
        0: "sss",           # steady-state selection
        1: "rws",           # roulette wheel selection
        2: "sus",           # stochastic universal selection
        3: "rank",          # rank selection
        4: "random",        # random selection
        5: "tournament"     # tournament selection
    }
    return parentSelectionDict[n]

def printGAParameters(GA_parameters, populationSize):
    pct_num_parents_mating = GA_parameters[0]
    parent_selection_type  = GA_parameters[1]
    keep_elitism           = GA_parameters[2]
    crossover_type         = GA_parameters[3]
    crossover_probability  = GA_parameters[4]
    pctAdaptiveInit        = GA_parameters[5]
    pctAdaptiveEnd         = GA_parameters[6]


    parent_selection_type = GA_getParentSelectionType(parent_selection_type)
    crossover_type        = GA_getCrossoverType(crossover_type)
    mutation_type         = "adaptive"
    keep_elitism          = round(keep_elitism)
    crossover_probability = round(crossover_probability, 2)
    num_parents_mating    = round(pct_num_parents_mating*populationSize, 3)
    mutation_num_genes    = (max(round(pctAdaptiveInit*populationSize), 1), max(round(pctAdaptiveEnd*pctAdaptiveInit*populationSize), 1))

    print(f"\nparent_selection_type: {parent_selection_type}\n"
          f"\tcrossover_type: {crossover_type}\n"
          f"\tmutation_type: {mutation_type}\n"
          f"\tkeep_elitism: {keep_elitism}\n"
          f"\tcrossover_probability: {crossover_probability}\n"
          f"\tnum_parents_mating: {num_parents_mating}\n"
          f"\tmutation_num_genes: {mutation_num_genes}")
    return

def printCromossome(cromossome):
    sigmaDEM                    = cromossome[0]
    sigmaREM                    = cromossome[1]
    windowSizeScaleDEM          = cromossome[2]
    windowSizeScaleREM          = cromossome[3]
    patchSizeDEM                = cromossome[4]
    patchSizeREM                = cromossome[5]
    nLevelsDEM                  = cromossome[6]
    nLevelsREM                  = cromossome[7]
    initLevelDEM                = cromossome[8]
    initLevelREM                = cromossome[9]
    scaleFactorDEM              = cromossome[10]
    scaleFactorREM              = cromossome[11]
    fastThresholdDEM            = cromossome[12]
    fastThresholdREM            = cromossome[13]

    sigmaDEM = int(sigmaDEM)
    sigmaREM = int(sigmaREM)
    windowSizeScaleDEM = int(windowSizeScaleDEM)
    windowSizeScaleREM = int(windowSizeScaleREM)
    patchSizeDEM = int(patchSizeDEM)
    patchSizeREM = int(patchSizeREM)
    nLevelsDEM = int(nLevelsDEM)
    nLevelsREM = int(nLevelsREM)
    initLevelDEM = int(initLevelDEM)
    initLevelREM = int(initLevelREM)
    scaleFactorDEM = round(scaleFactorDEM, 2)
    scaleFactorREM = round(scaleFactorREM, 2)
    fastThresholdDEM = int(fastThresholdDEM)
    fastThresholdREM = int(fastThresholdREM)

    print(f'''\nCromossome data:
        sigmaDEM: {sigmaDEM}
        sigmaREM: {sigmaREM}
        windowSizeScaleDEM: {windowSizeScaleDEM}
        windowSizeScaleREM: {windowSizeScaleREM}
        patchSizeDEM: {patchSizeDEM}
        patchSizeREM: {patchSizeREM}
        nLevelsDEM: {nLevelsDEM}
        nLevelsREM: {nLevelsREM}
        initLevelDEM: {initLevelDEM}
        initLevelREM: {initLevelREM}
        scaleFactorDEM: {scaleFactorDEM}
        scaleFactorREM: {scaleFactorREM}
        fastThresholdDEM: {fastThresholdDEM}
        fastThresholdREM: {fastThresholdREM}''')
    return

def printSample(sample):
    pitch        = sample[0]
    roll         = sample[1]
    x0           = sample[2]
    y0           = sample[3]
    subSize      = sample[4]
    rotAngle     = sample[5]
    resizeScale  = sample[6]
    noise        = sample[7]
    fileNumber   = sample[8]
    pitch       = round(pitch, 1)
    roll        = round(roll, 1)
    x0, y0      = round(x0, 2), round(y0, 2)
    subSize     = round(subSize, 2)
    rotAngle    = round(rotAngle, 1)
    resizeScale = round(resizeScale, 2)
    noise       = round(noise, 1)
    fileNumber  = int(fileNumber)
    file        = files[fileNumber]

    print(f'''
          Pitch: {pitch}
          Roll: {roll}
          x0, y0: ({x0}, {y0})
          subSize: {subSize}
          rotAngle: {rotAngle}
          resizeScale: {resizeScale}
          Noise: {noise}
          File: {file}''')
    return
    
def getMatcherType(WTA):
    return cv2.NORM_HAMMING*(WTA == 2) + cv2.NORM_HAMMING2*(WTA > 2)

def calculateNumFastFeatures(fastFeatureDensity, fastBlockSizeInPixels):
    return int(fastFeatureDensity*fastBlockSizeInPixels)

def correctData(number, column):
    if column == "sigmaImg" or column == "sigmaSub" or column == 'windowSizeScaleImg' or column == 'windowSizeScaleSub' or column == 'nLevelsImg' or column == 'nLevelsSub' or column == 'initLevelImg' or column == 'initLevelSub' or column == 'ratioImgAndSubFeatureNumber':
        return round(number)
    elif column == "scaleFactor":
        return round(number, 1)
    elif column == "patchSize" or column == "fastThreshold":
        number = round(number)
        if number % 2 == 0:
            return number + 1
        else: return number

def closestIndex(n, numArray):
    # Calcula as diferenças absolutas entre o número 'n' e os elementos do vetor 'numArray'
    diferenca_absoluta = np.abs(np.array(numArray) - n)
    
    # Encontra o índice do número mais próximo
    indice_numero_proximo = diferenca_absoluta.argmin()
    
    return indice_numero_proximo

def applyNoiseInCromossome(cromossome):
    numGenes = len(SPACE_ALL)
    numChangedGenes = int(abs(np.random.normal())) + 2
    selectedPositions = []
    while len(selectedPositions) <= numChangedGenes:
        posRandom = int(numGenes*np.random.random())
        if not posRandom in selectedPositions:
            selectedPositions.append(posRandom)

    for genePosition in selectedPositions:
        try:
            elementPosition = np.where(np.isclose(SPACE_ALL[genePosition], cromossome[genePosition]))[0][0]
        except:
            elementPosition = closestIndex(cromossome[genePosition], SPACE_ALL[genePosition])
        randomNumber = int(abs(np.random.normal()))
        if elementPosition == len(SPACE_ALL[genePosition])-1:
            elementPosition -= 1+randomNumber
        elif elementPosition == 0:
            elementPosition += 1+randomNumber
        else:
            elementPosition += (1+(np.random.rand() > 0.5)*-2)*(1+randomNumber)
            elementPosition = min(elementPosition, len(SPACE_ALL[genePosition])-1)
            elementPosition = max(elementPosition, 0)
    cromossome[genePosition] = SPACE_ALL[genePosition][elementPosition]
    return cromossome

def generateCentroidPopulation(popSize):
    popSize = int(popSize)
    maxClusters = 15
    filePreffix = 'Análises estatísticas\Cluster\Centroids\centroid_k_'
    fileSuffix = '.xlsx'
    remaining = popSize
    clusterNumberArray = []
    while remaining > 0:
        subtract = min(remaining, maxClusters)
        remaining -= subtract
        clusterNumberArray.append(subtract)

    populationArray = []
    for clusterNumber in clusterNumberArray:
        data = pd.read_excel(filePreffix + f'{clusterNumber}' + fileSuffix)
        columns = data.columns
        data = data.to_numpy()
        for i in range(len(data)):
            populationArray.append(data[i][:])
    
    for cromossome in populationArray:
        for geneIndex, col in enumerate(columns):
            cromossome[geneIndex] = correctData(cromossome[geneIndex], col)
        cromossome = applyNoiseInCromossome(cromossome)
    return populationArray

def selectSample(sampleId):
    '''
        self.x0             
        self.y0             
        self.x1             
        self.y1             
        self.pitch          
        self.roll           
        self.rotAngle       
        self.resizeScale    
        self.noiseIntensity 
    '''
    if sampleId == '3_samples':
        #sample_0 = [0.35, 0.5, 0.45, 0.6, 3, 2, 0.0, 0.7, 0.5, 2.0]
        #sample_1 = [0.3, 0.65, 0.4, 0.75, 4.0, 4.0, 140.0, 1.0, 0.5, 1.0]
        sample_2 = [0.5, 0.35, 0.65, 0.5, 2.0, 3.0, 70.0, 0.6, 1.5, 0.0]
        #samples  = [sample_0, sample_1, sample_2]
        samples = [sample_2]
    return samples

def GAParametersTableGen(numSamples, fileName = 'GA_parameters'):
    numParametersCombinations = numSamples
    GA_parametersPopulation = generatePopulation(numParametersCombinations, GA_PARAMETERS_SPACES)
    df = pd.DataFrame(data=GA_parametersPopulation, columns=GA_PARAMETERS_STRINGS)
    df.to_excel(f'{fileName}.xlsx', index=False)


def distortionDicToCromossome(sample):
    return [
        sample["x0"],
        sample["y0"],
        sample["x1"],
        sample["y1"],
        sample["pitch"],
        sample["roll"],
        sample["rotAngle"],
        sample["resizeScale"],
        sample["noiseIntensity"],
        sample["fileName"]
    ]

def generateGAExtensiveSearchSamples():
    sample1 = {
        "pitch": 4,
        "roll": 3,
        "rotAngle": 50,
        "x0": 0.65,
        "y0": 0.55,
        "x1": 0.75,
        "y1": 0.65,
        "resizeScale": 0.4,
        "noiseIntensity": 15,
        "fileName": r'Datasets\RS\merged_map.tif'
    }

    sample2 = {
        "pitch": 3,
        "roll": 5,
        "rotAngle": 260,
        "x0": 0.35,
        "y0": 0.35,
        "x1": 0.4,
        "y1": 0.4,
        "resizeScale": 0.5,
        "noiseIntensity": 2.5,
        "fileName": r'Datasets\Cyprus\Cyprus_data.tif'
    }

    sample3 = {
        "pitch": 1,
        "roll": 1,
        "rotAngle": 140,
        "x0": 0.5,
        "y0": 0.65,
        "x1": 0.7,
        "y1": 0.85,
        "resizeScale": 0.8,
        "noiseIntensity": 0.5,
        "fileName": r'Datasets\USGS\OK_Panhandle.tif'
    }

    # Convertendo os dados para vetores usando a função
    sample1_cromossome = distortionDicToCromossome(sample1)
    sample2_cromossome = distortionDicToCromossome(sample2)
    sample3_cromossome = distortionDicToCromossome(sample3)
    return [
        sample1_cromossome,
        sample2_cromossome,
        sample3_cromossome
    ]

if __name__ == '__main__':
    pass