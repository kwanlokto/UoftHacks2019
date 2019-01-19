def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][0] is predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0