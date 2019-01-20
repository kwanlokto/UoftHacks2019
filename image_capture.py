# Used https://www.youtube.com/watch?v=1XTqE7LFQjI
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
model = kaggle_CNN.load_model()

# Parameters
cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
threshold = 60  # BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

video = cv2.VideoCapture(0)
noBackground = 0


def removeBG(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


while True:
    #3. Create a frame object
    check, frame = video.read()

    #6. Converting to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.flip(gray, 1)  # flip the frame horizontally
    cv2.rectangle(gray, (gray.shape[1] - 560, 0),
                  (gray.shape[1], 560), (255, 0, 0), 2) #Display the rectangle on the right

    #4. Show the frame!
    cv2.imshow("Capturing", gray)

    #7. For playing
    key=cv2.waitKey(1)
    if noBackground:
        gray = removeBG(gray)
        #9. For getting only info from square region
        img = gray[0:560, gray.shape[1] - 560:gray.shape[1]]
        cv2.imshow("hand", img)
        small = cv2.resize(img, (0, 0), fx=0.05, fy=0.05)  # resize image to 240 x 240 px
        hand_pixels = small.flatten()
        if key == ord('a'):
            label = kaggle_CNN.run_model(model, np.array([hand_pixels]))
            print(alphabet[label])

    if key == ord('q'):
        break
    elif key == ord('b'):
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        noBackground = 1
#2. Shutdown the camera
video.release()

cv2.destroyAllWindows()



