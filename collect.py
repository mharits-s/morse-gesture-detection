import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 20
imgSize = 300
folder = "Data/pad"
counter = 0
max_img = 100
rec = False

if not os.path.exists(folder):
    os.makedirs(folder)
    print(f"Folder {folder} created.")

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Check if imgCrop is not empty
        if not imgCrop.size == 0:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCropShape = imgCrop.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord("s") and not rec:
        rec = True
        print("Recording started.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if rec:
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

        if counter >= max_img:
            print("Maximum number of images reached. Recording stopped.")
            rec = False
            break

# cap.release()
# cv2.destroyAllWindows()
