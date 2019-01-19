# Used https://www.youtube.com/watch?v=1XTqE7LFQjI
import k_NN
import checkModel
import readCSV
import cv2, time
import numpy as np

alphabet = {0: 'a', 1: 'b', 2:'c', 3: 'd', 4: 'e',
            5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j',
            10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o',
            15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't',
            20:'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z'}


def runModel(testPoint):
    trainingSet = readCSV.readCSV('./data_set/train.csv')
    k = 35
    neighbours = k_NN.getNearestNeighbours(trainingSet, testPoint, k)
    result = k_NN.getResponse(neighbours)
    return result


cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
#1. Create an object. Zero for external camera
video=cv2.VideoCapture(0)

#8. a variable
previous = None
a = 0
while True:
    a = a + 1

    #3. Create a frame object
    check, frame = video.read()
    # img = cv2.resize(frame, (28, 28))
    print(check)
    # print(frame) #Representing image

    #6. Converting to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.flip(gray, 1)  # flip the frame horizontally
    cv2.rectangle(gray, (gray.shape[1] - 480, 0),
                  (gray.shape[1], 480), (255, 0, 0), 2) #Display the rectangle on the right


    #4. Show the frame!
    cv2.imshow("Capturing", gray)

    #5. For press any key to out (miliseconds)
    # cv2.waitKey(0)

    #7. For playing
    key=cv2.waitKey(1)

    #9. For getting only info from square region

    img = gray[0:480, gray.shape[1] - 480:gray.shape[1]]
    cv2.imshow("hand", img)

    hand_pixels = np.array(img)
    hand_pixels.flatten()
    label = runModel(hand_pixels)

    if previous != label:
        a = 0
        previous = label

    # After 10 ish cycles and if the label is still the same then we can safely assume that
    # the label is accurate
    if a == 100:
        print(alphabet[label])
    if key == ord('q'):
        break

#2. Shutdown the camera
video.release()

cv2.destroyAllWindows()



