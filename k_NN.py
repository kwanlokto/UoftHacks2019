import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import operator



def getNearestNeighbours(trainingSet, testPoint, k):
    distances = []
    dict = {}
    for i in range(len(trainingSet)):
        dist = euclideanDistance(testPoint, trainingSet[i])
        if dist not in dict:
            dict[dist] = [trainingSet[i]]
        else:
            dict[dist].append(trainingSet[i])
        distances.append(dist)
        #distances.append((trainingSet[i], dist))
    distances_np_array = np.array(distances)
    distances_np_array.sort()
    # distances.sort(key=operator.itemgetter(1))
    neighbours = []
    x = 0
    size = 0
    while size < k:
        neighbours += dict[distances_np_array[x]]
        size = len(neighbours)
        x += 1
    return neighbours


def euclideanDistance(testPoint, trainPoint):
    test = np.array(testPoint)
    train = np.array(trainPoint)
    dist = np.linalg.norm(test - train)
    return dist


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][0]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1

    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

