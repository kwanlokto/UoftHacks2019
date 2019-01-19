import readCSV
import k_NN
import checkModel
import random

def runModel(testPoint):
    trainingSet = readCSV.readCSV('./data_set/train.csv')
    k = 35
    neighbours = k_NN.getNearestNeighbours(trainingSet, testPoint, k)
    result = k_NN.getResponse(neighbours)
    return result

def main():
    trainingSet = readCSV.readCSV('./data_set/train.csv')
    testSet = readCSV.readCSV('./data_set/test.csv')
    k = 60
    testSet = random.sample(testSet, 50)
    predictions = []
    for i in range(len(testSet)):
        neighbours = k_NN.getNearestNeighbours(trainingSet, testSet[i][1:], k)
        result = k_NN.getResponse(neighbours)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[i][0]))
    accuracy = checkModel.getAccuracy(testSet, predictions)
    print(accuracy)

main()
