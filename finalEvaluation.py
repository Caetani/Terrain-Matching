from geneticAlgorithm import GeneticAlgorithm
from terrainMatching import TerrainDistortion, TerrainMatching, EvaluateMatches
import pandas as pd
import numpy as np
import generalParameters as gp
import commom as cm
from datetime import datetime
from time import time


def setupGeneticAlgorithm(populationSize, numGenerations):
    gaParametersDict = {
        'pctNumParentsMating':  0.8,
        'parentSelectionType':  2,
        'keepElitism':          1,
        'crossoverType':        3,
        'crossoverProbability': 1,
        'pctAdaptiveInit':      0.2,
        'pctAdaptiveEnd':       0
    }
    gaParametersArray = [
        gaParametersDict['pctNumParentsMating'],
        gaParametersDict['parentSelectionType'],
        gaParametersDict['keepElitism'],
        gaParametersDict['crossoverType'],
        gaParametersDict['crossoverProbability'],
        gaParametersDict['pctAdaptiveInit'],
        gaParametersDict['pctAdaptiveEnd']
    ]
    geneticAlgorithm = GeneticAlgorithm(parametersArray=gaParametersArray)
    geneticAlgorithm.populationSize = populationSize
    geneticAlgorithm.numGenerations = numGenerations

    print(f"Genetic Algorithm Configuration")
    print(f"\tpopulationSize:         {geneticAlgorithm.populationSize}")
    print(f"\tnumGenerations:         {geneticAlgorithm.numGenerations}")
    print(f"\tparentSelectionType:    {geneticAlgorithm.parentSelectionType}")
    print(f"\tnumParentsMating:       {geneticAlgorithm.numParentsMating}")
    print(f"\tkeepElitism:            {geneticAlgorithm.keepElitism}")
    print(f"\tcrossoverType:          {geneticAlgorithm.crossoverType}")
    print(f"\tcrossoverProbability:   {geneticAlgorithm.crossoverProbability}")
    print(f"\tmutationNumGenes:       {geneticAlgorithm.mutationNumGenes}")
    
    return geneticAlgorithm

if __name__ == '__main__':
    firstSample, finalSample = 0, None
    resultFileName = 'Final_results'
    data = []


    distortionParametersData = pd.read_excel("Support files\Final Evaluation\Final_samples.xlsx")
    distortionParametersData.sort_values(by=['fileNumber'], ascending=True, inplace=True)
    if firstSample is not None and finalSample is not None:
        distortionParametersData = distortionParametersData[firstSample:finalSample]
    print(f"Using {len(distortionParametersData)} samples.")
    distortionParametersData = distortionParametersData.to_numpy()
    
    lastFileNumber = None
    for parameterCounter, distortionParameter in enumerate(distortionParametersData):
        fileNumber = int(distortionParameter[-1])
        if fileNumber != lastFileNumber:
            fileName = gp.files[fileNumber]
            noise = gp.datasetsNoises[fileNumber]
            distortionParameter[-1] = noise
            original_dem = cm.getMap(fileName, dtype=np.float32)

        terrainDistortion = TerrainDistortion(parametersArray=distortionParameter)
        terrainDistortion.distortElevationData(original_dem)

        print(f"\nSample {parameterCounter+1}/{len(distortionParametersData)}: {fileName}")
        print(f"\tx0:           {terrainDistortion.x0}")
        print(f"\ty0:           {terrainDistortion.y0}")
        print(f"\tx1:           {terrainDistortion.x1}")
        print(f"\ty1:           {terrainDistortion.y1}")
        print(f"\tpitch:        {terrainDistortion.pitch}")
        print(f"\troll:         {terrainDistortion.roll}")
        print(f"\tyaw:          {terrainDistortion.rotAngle}")
        print(f"\tresizeFactor: {terrainDistortion.resizeScale}")
        print(f"\tnoise:        {terrainDistortion.noiseIntensity}")

        geneticAlgorithm = setupGeneticAlgorithm(populationSize = 10,
                                                 numGenerations = 20)

        bestCromossome, bestFitness, bestFitnessFirstGen = geneticAlgorithm.runGenericAlgorithm(terrainDistortion=terrainDistortion)

        bestFitness = geneticAlgorithm.bestFitness   
        bestCromossome = geneticAlgorithm.bestCromossome

        terrainMatching = TerrainMatching(parametersArray=bestCromossome)
        terrainMatching.matchTerrains(DEM=terrainDistortion.DEM, REM=terrainDistortion.REM)

        evaluateMatches = EvaluateMatches(terrainMatchingObject=terrainMatching,
                                          terrainDistortionObject=terrainDistortion)
        evaluateMatches.evaluate()

        currentData = [
            terrainDistortion.x0,
            terrainDistortion.y0,
            terrainDistortion.x1,
            terrainDistortion.y1,
            terrainDistortion.pitch,
            terrainDistortion.roll,
            terrainDistortion.rotAngle,
            terrainDistortion.resizeScale,
            terrainDistortion.noiseIntensity,
            fileNumber,
            fileName,
            terrainMatching.sigmaDEM,
            terrainMatching.sigmaREM,
            terrainMatching.windowSizeScaleDEM,
            terrainMatching.windowSizeScaleREM,
            terrainMatching.patchSizeDEM,
            terrainMatching.patchSizeREM,
            terrainMatching.nLevelsDEM,
            terrainMatching.nLevelsREM,
            terrainMatching.initLevelDEM,
            terrainMatching.initLevelREM,
            terrainMatching.scaleFactorDEM,
            terrainMatching.scaleFactorREM,
            terrainMatching.fastThresholdDEM,
            terrainMatching.fastThresholdREM,
            terrainMatching.crossCheck,
            evaluateMatches.score,
            evaluateMatches.inliers,
            evaluateMatches.totalBestMatches
        ]
        data.append(currentData)

        terrainDistortion = None
        terrainMatching = None
        evaluateMatches = None
        geneticAlgorithm = None


    columns = [
        'terrainDistortion.x0',
        'terrainDistortion.y0',
        'terrainDistortion.x1',
        'terrainDistortion.y1',
        'terrainDistortion.pitch',
        'terrainDistortion.roll',
        'terrainDistortion.rotAngle',
        'terrainDistortion.resizeScale',
        'terrainDistortion.noiseIntensity',
        'fileNumber',
        'fileName',
        'terrainMatching.sigmaDEM',
        'terrainMatching.sigmaREM',
        'terrainMatching.windowSizeScaleDEM',
        'terrainMatching.windowSizeScaleREM',
        'terrainMatching.patchSizeDEM',
        'terrainMatching.patchSizeREM',
        'terrainMatching.nLevelsDEM',
        'terrainMatching.nLevelsREM',
        'terrainMatching.initLevelDEM',
        'terrainMatching.initLevelREM',
        'terrainMatching.scaleFactorDEM',
        'terrainMatching.scaleFactorREM',
        'terrainMatching.fastThresholdDEM',
        'terrainMatching.fastThresholdREM',
        'terrainMatching.crossCheck',
        'evaluateMatches.score',
        'evaluateMatches.inliers',
        'evaluateMatches.totalBestMatches'
    ]

    dataframe = pd.DataFrame(data, columns=columns)
    try:
        now = datetime.now()
        time = now.strftime("%Hh_%Mm_date_%d_%m_%Y")
        dataframe.to_excel(f"{resultFileName}_{time}.xlsx")
    except:
        print(f"Exeption in datetime generation.")
        dataframe.to_excel(f"{resultFileName}.xlsx")
