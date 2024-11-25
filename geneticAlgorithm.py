import pygad
from terrainMatching import TerrainMatching, EvaluateMatches
import commom as cm
import generalParameters as gp
import numpy as np
from random import shuffle
import concurrent.futures
import time

class GeneticAlgorithm():
    def __init__(self, parametersArray=None) -> None:
        ''' Independent variables '''
        self.parentSelectionType         = None
        self.numParentsMating            = None
        self.keepElitism                 = None
        self.crossoverType               = None
        self.crossoverProbability        = None
        self.mutationNumGenes            = None

        ''' Constants '''
        self.populationSize              = 10
        self.numGenerations              = 20
        self.solutionSpace               = gp.SPACE_ALL
        self.mutationType                = "adaptive"
        self.initialPopulationMultiplier = 5     # TODO Atualizar p 5   # Number of initial individuals testes before optimization
        self.initPopPctGoodInidividuals  = 0.25  # TODO Voltar para 0.25
        self.maxNumberOfTries            = 3
        self.maxNumCores                 = 3
        self.parallelProcessingConfig    = ["process", min(self.maxNumCores, self.populationSize)]
        self.showProgress                = True     # Used to observe the evolution of the algorithm.
        self.displayData                 = True     # Used to observe the behavior of the algorithm on a map.
        self.saveBestSolutions           = False
        self.saveSolutions               = False

        ''' Variables used in processing '''
        self.terrainDistortion           = None
        self.bestFitnessFirstGen         = None
        self.bestFitness                 = -np.inf
        self.bestCromossome              = None

        self.updateParameters(parametersArray)

    def updateParameters(self, parametersArray):
        pctNumParentsMating       = parametersArray[0]
        parentSelectionType       = parametersArray[1]
        keepElitism               = parametersArray[2]
        crossoverType             = parametersArray[3]
        crossoverProbability      = parametersArray[4]
        pctAdaptiveInit           = parametersArray[5]
        pctAdaptiveEnd            = parametersArray[6]

        ''' Avoding mismatches from differences like: 0.5000000001 != 0.5 '''
        pctNumParentsMating  = round(pctNumParentsMating, 3)
        crossoverProbability = round(crossoverProbability, 3)
        pctAdaptiveInit      = round(pctAdaptiveInit, 3)
        pctAdaptiveEnd       = round(pctAdaptiveEnd, 3)

        self.parentSelectionType  = gp.GA_getParentSelectionType(parentSelectionType)
        self.crossoverType        = gp.GA_getCrossoverType(crossoverType)
        self.numParentsMating     = round(pctNumParentsMating*self.populationSize)
        self.mutationNumGenes     = (max(round(pctAdaptiveInit*len(self.solutionSpace)), 1), max(round(pctAdaptiveEnd*len(self.solutionSpace)), 1))
        self.keepElitism          = round(keepElitism)
        self.crossoverProbability = round(crossoverProbability, 2)
        return

    def fitnessFunction(self, gaInstance, cromossome, solutionIndex):
        terrainMatching = TerrainMatching(parametersArray=cromossome)
        terrainMatching.matchTerrains(DEM=self.terrainDistortion.DEM, REM=self.terrainDistortion.REM)
        evaluateMatches = EvaluateMatches(terrainMatching, self.terrainDistortion)
        evaluateMatches.evaluate()
        fitness = evaluateMatches.score
        if self.showProgress: print(f"\t\tFitness = {fitness:_}\tInliers = {evaluateMatches.inliers} - bestMatches = {evaluateMatches.totalBestMatches}")
        return fitness

    def evaluatePopulation(self, population):
        fitnessArray = []
        for cromossome in population:
            fitness = self.fitnessFunction(None, cromossome, None)
            fitnessArray.append(fitness)
        assert len(fitnessArray) == len(population), "Error while evaluating population."
        return fitnessArray

    def generateInitialPopulation(self):
        numGoodInidividuals = max(int(self.initPopPctGoodInidividuals*self.populationSize), 1)
        initialPopulationSize = self.initialPopulationMultiplier*self.populationSize
        print(f"\nGenerating initial population...\n")
        goodCromossomes = []
        counter = 0
        bMinResults = False
        while not bMinResults and counter < self.maxNumberOfTries:
            print(f"Try {counter+1}/{self.maxNumberOfTries}")
            initialPopulation = cm.generatePopulation(initialPopulationSize, self.solutionSpace)
            dividedPopulation = np.array_split(initialPopulation, self.maxNumCores)
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.maxNumCores) as executor:
                populationScores = []
                fitnessArrays = executor.map(self.evaluatePopulation, dividedPopulation)
                for arr in fitnessArrays:
                    populationScores.append(np.array(arr))
                populationScores = np.concatenate(populationScores, axis=0).reshape(initialPopulationSize, 1)
                matrix = np.hstack((initialPopulation, populationScores))
                matrix = matrix[matrix[:, -1].argsort()][::-1]
                if matrix[numGoodInidividuals-1, -1] > 1:
                    bMinResults = True
                else:
                    i = 0
                    while matrix[i, -1] > 1:
                        goodCromossomes.append(matrix[i, :])
                        i += 1
                    counter += 1
        if not bMinResults:
            if len(goodCromossomes) > self.populationSize:
                goodCromossomes = goodCromossomes[goodCromossomes[:, -1].argsort()][::-1]
                result = goodCromossomes
            elif len(goodCromossomes) == self.populationSize:
                result = goodCromossomes
            else:
                if len(goodCromossomes) > 0:
                    goodCromossomes = np.delete(goodCromossomes, [-1], axis=1)
                    popComplement = cm.generatePopulation(self.populationSize-len(goodCromossomes), self.solutionSpace)
                    result = np.vstack((goodCromossomes, popComplement))
                    return result
                else:
                    result = shuffle(initialPopulation)
        print(f"Inial population generated.\n")
        result = np.delete(matrix, [-1], axis=1)
        return result[:self.populationSize]

    def callbackGeneration(self, ga_instance):
        generation = ga_instance.generations_completed
        print(f"\n\t[Generation: {generation}]")
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        if generation == 0: self.bestFitnessFirstGen = solution_fitness
        if solution_fitness > self.bestFitness:
            self.bestFitness = solution_fitness
            self.bestCromossome = solution
        try:
            print(f"\tFitness = {solution_fitness:_}\n\tSolution:", end='')
            gp.printCromossome(solution)
            print()
        except:
            pass

    def runGenericAlgorithm(self, terrainDistortion):
        self.terrainDistortion = terrainDistortion
        initialPopulation = self.generateInitialPopulation()
        gaInstance = pygad.GA(num_generations       = self.numGenerations, 
                              num_parents_mating    = self.numParentsMating,
                              gene_space            = self.solutionSpace,
                              initial_population    = initialPopulation,
                              fitness_func          = self.fitnessFunction,
                              parent_selection_type = self.parentSelectionType,
                              keep_elitism          = self.keepElitism,
                              crossover_type        = self.crossoverType,
                              crossover_probability = self.crossoverProbability,
                              mutation_type         = self.mutationType,
                              mutation_num_genes    = self.mutationNumGenes,
                              save_best_solutions   = self.saveBestSolutions,
                              save_solutions        = self.saveSolutions,
                              parallel_processing   = self.parallelProcessingConfig,
                              on_generation         = self.callbackGeneration,
                              on_start              = self.callbackGeneration)
        gaInstance.run()
        while gaInstance.run_completed != True: time.sleep(0.1)
        print(f"\n\tBest Solution Fitness = {self.bestFitness:_}")
        gp.printCromossome(self.bestCromossome)
        return self.bestCromossome, self.bestFitness, self.bestFitnessFirstGen