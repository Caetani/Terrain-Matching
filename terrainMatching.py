import commom as cm
from commom import float32_to_uint8_GPU # Cannot use cm. prefix due to @vectorize, don't know why.
import generalParameters as gp
import numpy as np
import cv2
from math import sqrt

class TerrainMatching:
    def __init__(self, parametersArray=None) -> None:
        ''' Independent variables '''
        self.sigmaDEM               = None
        self.sigmaREM               = None
        self.windowSizeScaleDEM     = None
        self.windowSizeScaleREM     = None
        self.patchSizeDEM           = None
        self.patchSizeREM           = None
        self.nLevelsDEM             = None
        self.nLevelsREM             = None
        self.firstLevelDEM          = None
        self.firstLevelREM          = None
        self.scaleFactorDEM         = None
        self.scaleFactorREM         = None
        self.fastThresholdDEM       = None
        self.fastThresholdREM       = None
        self.crossCheck             = None

        ''' Dependent variables '''
        self.edgeThresholdDEM       = None
        self.edgeThresholdREM       = None

        ''' Constants '''
        self.numFeatures            = 500
        self.numMatches             = 100
        self.numBestMatches         = 10
        self.WTA                    = 2
        self.blockSizeInPixels      = 1_000
        self.matcherType            = cv2.NORM_HAMMING
        self.scoreType              = cv2.ORB_HARRIS_SCORE

        ''' Outputs '''
        self.matches                = None
        self.keypointsDEM           = None
        self.keypointsREM           = None

        self.updateParameters(parametersArray)

    def updateParameters(self, parametersArray):
        sigmaDEM                    = parametersArray[0]
        sigmaREM                    = parametersArray[1]
        windowSizeScaleDEM          = parametersArray[2]
        windowSizeScaleREM          = parametersArray[3]
        patchSizeDEM                = parametersArray[4]
        patchSizeREM                = parametersArray[5]
        nLevelsDEM                  = parametersArray[6]
        nLevelsREM                  = parametersArray[7]
        initLevelDEM                = parametersArray[8]
        initLevelREM                = parametersArray[9]
        scaleFactorDEM              = parametersArray[10]
        scaleFactorREM              = parametersArray[11]
        fastThresholdDEM            = parametersArray[12]
        fastThresholdREM            = parametersArray[13]
        crossCheck                  = parametersArray[14]

        self.sigmaDEM              = sigmaDEM
        self.sigmaREM              = sigmaREM
        self.windowSizeScaleDEM    = windowSizeScaleDEM
        self.windowSizeScaleREM    = windowSizeScaleREM

        self.patchSizeDEM          = int(patchSizeDEM)
        self.patchSizeREM          = int(patchSizeREM)
        self.nLevelsDEM            = int(nLevelsDEM)
        self.nLevelsREM            = int(nLevelsREM)
        self.initLevelDEM          = int(initLevelDEM)
        self.initLevelREM          = int(initLevelREM)
        self.scaleFactorDEM        = round(scaleFactorDEM, 2)
        self.scaleFactorREM        = round(scaleFactorREM, 2)
        self.fastThresholdDEM      = int(fastThresholdDEM)
        self.fastThresholdREM      = int(fastThresholdREM)
        self.crossCheck            = bool(crossCheck)

        self.edgeThresholdDEM      = self.patchSizeDEM
        self.edgeThresholdREM      = self.patchSizeREM

        return

    def matchTerrains(self, DEM, REM):   # TODO DEM and REM are copies (.copy()) of the original models.
        ''' Preprocessing '''
        elevationPreprocessing = ElevationPreprocessing(self)
        DEM, REM = elevationPreprocessing.preprocess(DEM=DEM, REM=REM)

        ''' Feature matching '''
        featureMatching = FeatureMatching(self)
        self.matches, self.keypointsDEM, self.keypointsREM = featureMatching.matchFeatures(DEM, REM)
        return

class ElevationPreprocessing:
    def __init__(self, terrainMatchingObject) -> None:
        self.TMParameters = terrainMatchingObject
        self.windowSizeDEM = int(self.TMParameters.windowSizeScaleDEM*self.TMParameters.sigmaDEM)
        self.windowSizeREM = int(self.TMParameters.windowSizeScaleREM*self.TMParameters.sigmaREM)

        if self.windowSizeDEM % 2 == 0: self.windowSizeDEM = self.windowSizeDEM + 1
        if self.windowSizeREM % 2 == 0: self.windowSizeREM = self.windowSizeREM + 1
        
    def preprocess(self, DEM, REM):
        DEM = self.gradientMagnitude(elevationData=DEM,
                                     windowsSize=self.windowSizeDEM,
                                     sigma=self.TMParameters.sigmaDEM)
        REM = self.gradientMagnitude(elevationData=REM,
                                     windowsSize=self.windowSizeREM,
                                     sigma=self.TMParameters.sigmaREM)
        DEM, REM = self.normalizeElevationData(DEM, REM)
        return DEM, REM

    def gradientMagnitude(self, elevationData, windowsSize, sigma):
        elevationData = cv2.GaussianBlur(elevationData, (windowsSize, windowsSize), sigmaX=sigma, sigmaY=sigma)
        gradX = cv2.Sobel(elevationData, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
        gradY = cv2.Sobel(elevationData, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
        elevationData = cv2.magnitude(gradX,gradY)
        del gradX, gradY
        return elevationData

    def normalizeElevationData(self, DEM, REM):
        maxValue = np.float32(max(np.max(DEM), np.max(REM)))
        minValue = np.float32(min(np.min(DEM), np.min(REM)))
        DEM = float32_to_uint8_GPU(DEM, minValue, maxValue)
        REM = float32_to_uint8_GPU(REM, minValue, maxValue)
        return DEM, REM

class FeatureMatching:
    def __init__(self, terrainMatchingObject) -> None:
        self.TMParameters = terrainMatchingObject

    def matchFeatures(self, DEM, REM):
        try:
            nBlocksDEM = cm.calculateNumberOfBlocks(lenghtInPixels=self.TMParameters.blockSizeInPixels, img=DEM)
            keypointsDEM, descriptorsDEM = self.computeSegmentedORB(elevationData=DEM,
                                                                    nBlocks=nBlocksDEM,
                                                                    numFeatures=self.TMParameters.numFeatures,
                                                                    nLevels=self.TMParameters.nLevelsDEM,
                                                                    firstLevel=self.TMParameters.firstLevelDEM,
                                                                    scaleFactor=self.TMParameters.scaleFactorDEM,
                                                                    WTA=self.TMParameters.WTA,
                                                                    edgeThreshold=self.TMParameters.edgeThresholdDEM,
                                                                    scoreType=self.TMParameters.scoreType,
                                                                    patchSize=self.TMParameters.patchSizeDEM,
                                                                    fastThreshold=self.TMParameters.fastThresholdDEM)
            keypointsREM, descriptorsREM = self.computeORB(elevationData=REM,
                                                        numFeatures=self.TMParameters.numFeatures,
                                                        nLevels=self.TMParameters.nLevelsREM,
                                                        firstLevel=self.TMParameters.firstLevelREM,
                                                        scaleFactor=self.TMParameters.scaleFactorREM,
                                                        WTA=self.TMParameters.WTA,
                                                        edgeThreshold=self.TMParameters.edgeThresholdREM,
                                                        scoreType=self.TMParameters.scoreType,
                                                        patchSize=self.TMParameters.patchSizeREM,
                                                        fastThreshold=self.TMParameters.fastThresholdREM)
            matches = self.matchDescriptors(descriptorsDEM, descriptorsREM)
            result = matches, keypointsDEM, keypointsREM
        except:
            result = None, None, None

        return result

    def computeSegmentedORB(self, elevationData, nBlocks, numFeatures, nLevels, firstLevel, scaleFactor, WTA, edgeThreshold, scoreType, patchSize, fastThreshold):
        orb = cv2.ORB.create(nfeatures=numFeatures,
                             nlevels=nLevels,
                             firstLevel=firstLevel,
                             scaleFactor=scaleFactor,
                             WTA_K=WTA,
                             edgeThreshold=edgeThreshold,
                             scoreType=scoreType,
                             patchSize=patchSize,
                             fastThreshold=fastThreshold)
        segmentedElevationDataKeypoints = cm.segmentedORBDetect(elevationData, orb, nBlocks=nBlocks)
        keypoints, descriptors = orb.compute(elevationData, keypoints=segmentedElevationDataKeypoints)
        orb = None
        del segmentedElevationDataKeypoints
        return keypoints, descriptors

    def computeORB(self, elevationData, numFeatures, nLevels, firstLevel, scaleFactor, WTA, edgeThreshold, scoreType, patchSize, fastThreshold):
        orb = cv2.ORB.create(nfeatures=numFeatures,
                             scaleFactor=scaleFactor,
                             nlevels=nLevels,
                             firstLevel=firstLevel,
                             WTA_K=WTA,
                             edgeThreshold=edgeThreshold,
                             scoreType=scoreType,
                             patchSize=patchSize,
                             fastThreshold=fastThreshold)
        keypoints, descriptors = orb.detectAndCompute(elevationData, None)
        orb = None
        return keypoints, descriptors

    def matchDescriptors(self, descriptorsDEM, descriptorsREM):
        bf = cv2.BFMatcher(self.TMParameters.matcherType, crossCheck=self.TMParameters.crossCheck)                              
        matches = bf.match(descriptorsDEM, descriptorsREM)
        bf = None
        matches = sorted(matches, key=lambda x:x.distance)
        matches = matches[:self.TMParameters.numMatches]
        return matches

class TerrainDistortion:
    def __init__(self, parametersArray=None) -> None:
        ''' Independent variables'''     # Unit
        self.resizeScale          = None # [px^2/px^2]
        self.x0                   = None # [%/%]
        self.y0                   = None # [%/%]
        self.x1                   = None # [%/%]
        self.y1                   = None # [%/%]
        self.rotAngle             = None # [degrees]
        self.pitch                = None # [degrees]
        self.roll                 = None # [degrees]
        self.noiseIntensity       = None # [meters]

        ''' Dependent variables '''
        self.cx                   = None # [px]
        self.cy                   = None # [px]
        self.absoluteX0           = None # [px]
        self.absoluteY0           = None # [px]
        self.absoluteX1           = None # [px]
        self.absoluteY1           = None # [px]

        ''' Variables used for processing '''
        self.DEM                  = None
        self.REM                  = None
        self.sigmaDEM             = None
        self.sigmaREM             = None

        self.updateParameters(parametersArray)

    def updateParameters(self, parametersArray):
        self.x0                   = parametersArray[0]
        self.y0                   = parametersArray[1]
        self.x1                   = parametersArray[2]
        self.y1                   = parametersArray[3]
        self.pitch                = parametersArray[4]
        self.roll                 = parametersArray[5]
        self.rotAngle             = parametersArray[6]
        self.resizeScale          = parametersArray[7]
        self.noiseIntensity       = parametersArray[8]

    def distortElevationData(self, elevationData):
        self.generateREM(elevationData)
        self.generateDEM(elevationData)
        return

    def generateREM(self, elevationData):
        height, width = elevationData.shape
        self.absoluteX0, self.absoluteX1 = round(self.x0*width), round(self.x1*width)
        self.absoluteY0, self.absoluteY1 = round(self.y0*height), round(self.y1*height)
        localCx, localCy = round(width/2), round(height/2)
        rotatedPts = cm.rotate2DRectangle(self.absoluteX0, self.absoluteX1,
                                          self.absoluteY0, self.absoluteY1,
                                          cx=localCx, cy=localCy,
                                          deg=-self.rotAngle)
        for p in rotatedPts:
            x_p, y_p = p
            assert x_p >= 0 and x_p <= width and y_p >= 0 and y_p <= height, "Points out of range."
        elevationData = cm.rotateMap(elevationData, self.rotAngle)
        elevationData = cm.cutSubregion(map=elevationData,
                                        x0=self.absoluteX0, y0=self.absoluteY0,
                                        x1=self.absoluteX1, y1=self.absoluteY1)
        self.sigmaREM = np.std(elevationData)
        elevationData = cm.applyPlaneDistortion(elevationData, self.pitch, self.roll)
        elevationData = cm.applyNoise(elevationData, self.noiseIntensity, dtype=np.float32)
        self.REM = elevationData
        return

    def generateDEM(self, elevationData):
        resizedHeight, resizedWidth = elevationData.shape
        resizedHeight, resizedWidth = round(self.resizeScale*resizedHeight), round(self.resizeScale*resizedWidth)
        self.sigmaDEM = np.std(elevationData)
        if self.resizeScale < 1:
            elevationData = cv2.resize(elevationData, dsize=(resizedWidth, resizedHeight), interpolation=cv2.INTER_AREA)
        elif self.resizeScale > 1:
            elevationData = cv2.resize(elevationData, dsize=(resizedWidth, resizedHeight), interpolation=cv2.INTER_CUBIC)
        self.absoluteX0, self.absoluteX1 = self.absoluteX0*self.resizeScale, self.absoluteX1*self.resizeScale
        self.absoluteY0, self.absoluteY1 = self.absoluteY0*self.resizeScale, self.absoluteY1*self.resizeScale
        self.cx = round(resizedWidth/2)
        self.cy = round(resizedHeight/2)
        self.DEM = elevationData
        return

class EvaluateMatches:
    def __init__(self, terrainMatchingObject, terrainDistortionObject) -> None:
        ''' Independent variables '''
        self.terrainMatching   = terrainMatchingObject
        self.terrainDistortion = terrainDistortionObject

        ''' Dependent variables '''
        self.score            = None
        self.inliers          = None
        self.totalBestMatches = None
        self.thresholdRadius  = max(1, sqrt(2*((1/terrainDistortionObject.resizeScale)-1)**2)+(1/terrainDistortionObject.resizeScale))

        ''' Contants '''
        self.bestMatches            = 10
        self.rectangleDrawOffset    = 5
        self.shapeColor             = (0, 255, 0)
        self.flags                  = 2
        self.matchesThickness       = 30
        self.showMatchesResizeRatio = 0.1
        self.defaultNullScore       = 10**-3
        self.defaultFailureScore    = 10**-9

    def evaluate(self):
        if self.terrainMatching.matches and self.terrainMatching.keypointsDEM and self.terrainMatching.keypointsREM:
            self.score, self.inliers, self.totalBestMatches = cm.computeMatchesScore(
                                                                    self.terrainMatching.matches,
                                                                    self.terrainMatching.keypointsDEM,
                                                                    self.terrainMatching.keypointsREM,
                                                                    self.terrainDistortion.resizeScale,
                                                                    self.terrainDistortion.absoluteX0,
                                                                    self.terrainDistortion.absoluteY0,
                                                                    self.terrainDistortion.cx,
                                                                    self.terrainDistortion.cy,
                                                                    self.terrainDistortion.rotAngle,
                                                                    thresholdRadius=self.thresholdRadius,
                                                                    matcherType=self.terrainMatching.matcherType,
                                                                    pitch=self.terrainDistortion.pitch,
                                                                    roll=self.terrainDistortion.roll,
                                                                    numBestMatches=self.bestMatches)
            if self.score == 0: self.score = self.defaultNullScore
        else:
            self.score              = self.defaultFailureScore
            self.inliers            = 0
            self.totalBestMatches   = 0
        return
    
    def showMatches(self, saveImage=None):
        DEM = self.terrainDistortion.DEM
        REM = self.terrainDistortion.REM
        if self.terrainMatching.keypointsDEM and self.terrainMatching.keypointsREM:
            rectangleCorners = cm.rotate2DRectangle(self.terrainDistortion.absoluteX0,
                                                    self.terrainDistortion.absoluteX1,
                                                    self.terrainDistortion.absoluteY0,
                                                    self.terrainDistortion.absoluteY1,
                                                    self.terrainDistortion.cx,
                                                    self.terrainDistortion.cy,
                                                    -self.terrainDistortion.rotAngle)
            DEMtoDisplay = float32_to_uint8_GPU(DEM, np.min(DEM), np.max(DEM))
            DEMtoDisplay = cv2.cvtColor(DEMtoDisplay, cv2.COLOR_GRAY2BGR)
            DEMtoDisplay = cm.drawPolygon(DEMtoDisplay, rectangleCorners, color=self.shapeColor)
            heightREM, widthREM = REM.shape
            REMtoDisplay = float32_to_uint8_GPU(REM, np.min(REM), np.max(REM))
            REMtoDisplay = cm.drawRectangle(REMtoDisplay,
                                            self.rectangleDrawOffset,
                                            self.rectangleDrawOffset,
                                            widthREM-self.rectangleDrawOffset,
                                            heightREM-self.rectangleDrawOffset)
            finalDisplay = cv2.drawMatches(DEMtoDisplay,
                                        self.terrainMatching.keypointsDEM,
                                        REMtoDisplay,
                                        self.terrainMatching.keypointsREM,
                                        self.terrainMatching.matches[:self.bestMatches],
                                        None,
                                        flags=self.flags,
                                        matchesThickness=self.matchesThickness)
            finalDisplay = cm.resizeImage(finalDisplay, self.showMatchesResizeRatio)
            
            print(f"\nInliers = {self.inliers}\nBest matches = {self.totalBestMatches}\n\n")
            finalDisplay[np.all(finalDisplay == [0, 0, 0], axis=-1)] = [255, 255, 255]
            cm.showImage(finalDisplay)
            if saveImage:
                cv2.imwrite(saveImage, finalDisplay)
        else:
            print(f"No valid data to be displayed.")
        return

    def showSubregion(self, fileName=None):
        rectangleCorners = cm.rotate2DRectangle(self.terrainDistortion.absoluteX0,
                                                self.terrainDistortion.absoluteX1,
                                                self.terrainDistortion.absoluteY0,
                                                self.terrainDistortion.absoluteY1,
                                                self.terrainDistortion.cx,
                                                self.terrainDistortion.cy,
                                                -self.terrainDistortion.rotAngle)
        DEMtoDisplay = cm.float32_to_uint8_CPU(self.terrainDistortion.DEM,
                                               np.min(self.terrainDistortion.DEM),
                                               np.max(self.terrainDistortion.DEM))
        DEMtoDisplay = cv2.cvtColor(DEMtoDisplay, cv2.COLOR_GRAY2BGR)
        DEMtoDisplay = cm.drawPolygon(DEMtoDisplay, rectangleCorners, color=self.shapeColor)
        finalDisplay = cm.resizeImage(DEMtoDisplay, self.showMatchesResizeRatio)
        cm.showImage(finalDisplay)
        if fileName:
            print(f"Saving image...")
            cv2.imwrite(f"{fileName}.png", finalDisplay)

if __name__ == '__main__':
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
    distortionParameters = [0.3, 0.4, 0.45, 0.55, 2, 1, 45, 0.4, 10]
    elevationData = cm.getMap(gp.files[0], dtype=np.float32)
    terrainDistortion = TerrainDistortion(parametersArray=distortionParameters)
    terrainDistortion.distortElevationData(elevationData)
    DEM, REM = terrainDistortion.DEM, terrainDistortion.REM
    '''
        self.sigmaDEM               = None
        self.sigmaREM               = None
        self.windowSizeScaleDEM     = None
        self.windowSizeScaleREM     = None
        self.patchSizeDEM           = None
        self.patchSizeREM           = None
        self.nLevelsDEM             = None
        self.nLevelsREM             = None
        self.firstLevelDEM          = None
        self.firstLevelREM          = None
        self.scaleFactorDEM         = None
        self.scaleFactorREM         = None
        self.fastThresholdDEM       = None
        self.fastThresholdREM       = None
    '''
    terrainMatching = TerrainMatching(parametersArray=[2, 2, 2, 2, 31, 31, 8, 8, 0, 0, 1.2, 1.2, 20, 20, 1])
    terrainMatching.matchTerrains(DEM, REM)
    evaluateMatches = EvaluateMatches(terrainMatching, terrainDistortion)
    evaluateMatches.evaluate()
    evaluateMatches.showMatches(DEM, REM)