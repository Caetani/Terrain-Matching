import commom as cm
import generalParameters as gp
from terrainMatching import TerrainDistortion, TerrainMatching, EvaluateMatches



samples = gp.generateGAExtensiveSearchSamples()

for i, sample in enumerate(samples):
    fileName = sample.pop(-1)
    dem = cm.getMap(fileName)
    terrainDistortion = TerrainDistortion(parametersArray=sample)
    terrainDistortion.distortElevationData(elevationData=dem)
    evaluateMatches = EvaluateMatches(None, terrainDistortionObject=terrainDistortion)
    evaluateMatches.shapeColor = (255, 6, 243)
    evaluateMatches.showSubregion(fileName=f"{i}_rosa")