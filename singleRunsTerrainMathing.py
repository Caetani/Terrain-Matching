from geneticAlgorithm import GeneticAlgorithm
from terrainMatchingDebug import TerrainDistortion, TerrainMatching, EvaluateMatches
import pandas as pd
import numpy as np
import generalParameters as gp
import commom as cm
from datetime import datetime
from time import time

''' 
This scripts was created to generate images of sucessful and unsucessful matches
to be shown in the documentation.
In addition, it also creates images showing the effect of the distorion process.
'''


if __name__ == '__main__':
    terrainDistortionAttrs = [
        'x0',            
        'y0',            
        'x1',            
        'y1',            
        'pitch',         
        'roll',          
        'rotAngle',      
        'resizeScale',   
        'noiseIntensity'
    ]
    terrainMatchingAttrs = [
        'sigmaDEM.1',          
        'sigmaREM.1',          
        'windowSizeScaleDEM',
        'windowSizeScaleREM',
        'patchSizeDEM',      
        'patchSizeREM',      
        'nLevelsDEM',        
        'nLevelsREM',        
        'initLevelDEM',     
        'initLevelREM',     
        'scaleFactorDEM',    
        'scaleFactorREM',    
        'fastThresholdDEM', 
        'fastThresholdREM',  
        'crossCheck'        
    ]

    data = pd.read_excel("Results\Final Evaluation\Final Results.xlsx")
    data.sort_values(['score'], ascending=False, inplace=True)
    bestSamples = data.groupby(by=['fileNumber']).first()

    saves = ['rs', 'cy', 'ok']

    for i in range(len(bestSamples)):
        fileName = bestSamples['fileName'].iloc[i]
        elevationData = cm.getMap(fileName, dtype=np.float32)
        height, width = elevationData.shape

        elevationData[elevationData == 0] = 5

        terrainDistortionCromossome = []
        terrainMatchingCromossome = []

        for attr in terrainDistortionAttrs:
            terrainDistortionCromossome.append(bestSamples[attr].iloc[i])
        for attr in terrainMatchingAttrs:
            terrainMatchingCromossome.append(bestSamples[attr].iloc[i])
        
        print(terrainMatchingCromossome)
        if True:
            print(terrainDistortionCromossome)
            terrainDistortionCromossome = [0.3, 0.3, 0.6, 0.6, 0, 0, 0, 1, 0]
            terrainDistortion = TerrainDistortion(parametersArray=terrainDistortionCromossome)
            terrainDistortion.distortElevationData(elevationData)

            cm.scaledDepthMap(terrainDistortion.REM)

            terrainMatching = TerrainMatching(parametersArray=terrainMatchingCromossome)
            terrainMatching.matchTerrains(DEM=terrainDistortion.DEM, REM=terrainDistortion.REM)

            evaluateMatches = EvaluateMatches(terrainMatchingObject=terrainMatching,
                                            terrainDistortionObject=terrainDistortion)
            evaluateMatches.evaluate()
            evaluateMatches.showMatches(saveImage=f"{saves[i]}.jpg")