import readCSV
import k_NN
import checkModel


def main():
    trainingSet = readCSV.readCSV('./data_set/train.csv')
    testSet = readCSV.readCSV('./data_set/test.csv')
    k = 35

    predictions = []
    for i in range(len(testSet)):
        neighbours = k_NN.getNearestNeighbours(trainingSet, testSet[i], k)
        result = k_NN.getResponse(neighbours)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[i][0]))
    accuracy = checkModel.getAccuracy(testSet, predictions)
    print(accuracy)

main()
