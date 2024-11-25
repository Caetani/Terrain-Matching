from geneticAlgorithm import GeneticAlgorithm
from terrainMatching import TerrainDistortion
import pandas as pd
import commom as cm
import generalParameters as gp
from datetime import datetime
from time import time

numExecutionsPerHiperparameter = 5
firstParameter, finalParameter = 0, 5
resultFileName = f'GA_parameters_grid_search_{firstParameter}-{finalParameter}'

if __name__ == '__main__':
    t0 = time()
    elevationSamplesParameters = gp.generateGAExtensiveSearchSamples()
    gaParametersTable = pd.read_excel('Support files\GA Parameters Optimization\GA_parameters_bestParametersAnalysis.xlsx').to_numpy()

    gaParametersTable = gaParametersTable[firstParameter:finalParameter] # TODO Delete this line
    
    ''' Creating history arrays for each execution '''
    histBestFitness          = []
    histBestFitnessFirstGen  = []
    histDeltaFitness         = []
    histSampleNumber         = []
    histFileName             = []
    histGaParameterCounter   = []
    histParentSelectionType  = []
    histCrossoverType        = []
    histNumParentsMating     = []
    histMutationNumGenes     = []
    histKeepElitism          = []
    histCrossoverProbability = []

    for sampleCounter, sampleParameters in enumerate(elevationSamplesParameters, start=1):
        fileName = sampleParameters.pop(-1)
        elevationData = cm.getMap(fileName, dtype=np.float32)
        terrainDistortion = TerrainDistortion(parametersArray=sampleParameters)
        terrainDistortion.distortElevationData(elevationData)
        for gaParameterCounter, gaParameters in enumerate(gaParametersTable, start=firstParameter):
            for execution in range(1, numExecutionsPerHiperparameter+1):
                geneticAlgorithm = GeneticAlgorithm(parametersArray=gaParameters)
                bestCromossome, bestFitness, bestFitnessFirstGen = geneticAlgorithm.runGenericAlgorithm(terrainDistortion=terrainDistortion)

                ''' Populate history arrays '''
                histBestFitness.append(bestFitness)
                histBestFitnessFirstGen.append(bestFitnessFirstGen)
                histDeltaFitness.append(bestFitness-bestFitnessFirstGen)
                histSampleNumber.append(sampleCounter)
                histFileName.append(fileName)
                histGaParameterCounter.append(gaParameterCounter)
                histParentSelectionType.append(geneticAlgorithm.parentSelectionType)
                histCrossoverType.append(geneticAlgorithm.crossoverType)
                histNumParentsMating.append(geneticAlgorithm.numParentsMating)
                histMutationNumGenes.append(geneticAlgorithm.mutationNumGenes)
                histKeepElitism.append(geneticAlgorithm.keepElitism)
                histCrossoverProbability.append(geneticAlgorithm.crossoverProbability)
            
    d = {
        'ParameterCounter': histGaParameterCounter,
        'ParentSelectionType': histParentSelectionType,
        'CrossoverType': histCrossoverType,
        'NumParentsMating': histNumParentsMating,
        'MutationNumGenes': histMutationNumGenes,
        'KeepElitism': histKeepElitism,
        'CrossoverProbability': histCrossoverProbability,
        'SampleNumber': histSampleNumber,
        'FileName': histFileName,
        'bestFitness': histBestFitness,
        'bestFitnessFirstGen': histBestFitnessFirstGen,
        'deltaFitness': histDeltaFitness
    }

    resultsDataFrame = pd.DataFrame(data=d)
    try:
        now = datetime.now()
        time = now.strftime("%Hh_%Mm_date_%d_%m_%Y")
        resultsDataFrame.to_excel(f"{resultFileName}_{time}.xlsx")
    except:
        print(f"Exeption in datetime generation.")
        resultsDataFrame.to_excel(f"{resultFileName}.xlsx")
    
    #print(f"\nData saved in Excel file. End of execution.\nTotal time = {time()-t0} seconds.")