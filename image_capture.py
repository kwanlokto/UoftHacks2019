# Used https://www.youtube.com/watch?v=1XTqE7LFQjI
import k_NN
import checkModel
import readCSV
import cv2
import kaggle_CNN
import numpy as np

alphabet = {0: 'a', 1: 'b', 2:'c', 3: 'd', 4: 'e',
            5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j',
            10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o',
            15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't',
            20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z'}

trainingSet = readCSV.readCSV('./data_set/train.csv')
trainingPoints = [i[1:] for i in trainingSet]
trainingLabels = [i[0] for i in trainingSet]
model = kaggle_CNN.load_model()

cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
video = cv2.VideoCapture(0)

while True:
    #3. Create a frame object
    check, frame = video.read()

    # print(frame) #Representing image

    #6. Converting to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.flip(gray, 1)  # flip the frame horizontally
    cv2.rectangle(gray, (gray.shape[1] - 560, 0),
                  (gray.shape[1], 560), (255, 0, 0), 2) #Display the rectangle on the right

    #4. Show the frame!
    cv2.imshow("Capturing", gray)

    #7. For playing
    key=cv2.waitKey(1)

    #9. For getting only info from square region
    img = gray[0:560, gray.shape[1] - 560:gray.shape[1]]
    cv2.imshow("hand", img)
    small = cv2.resize(img, (0, 0), fx=0.05, fy=0.05)  # resize image to 240 x 240 px
    hand_pixels = small.flatten()
    if key == ord('q'):
        break
    elif key == ord('a'):
        label = kaggle_CNN.run_model(model, np.array([hand_pixels]))
        # label = run_model(hand_pixels)
        print(alphabet[label])

#2. Shutdown the camera
video.release()

cv2.destroyAllWindows()



