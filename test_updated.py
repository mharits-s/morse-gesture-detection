import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
from gtts import gTTS
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
labels = ["dash", "dit", "enter", "new", "next", "space"]

# Inisialisasi variabel untuk menyimpan pola-pola
current_pattern = []
last_detected_gesture = None 
last_detection_time = time.time()


while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

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
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            if time.time() - last_detection_time > 4:
                # Menyimpan pola ke dalam variabel current_pattern
                if labels[index] != last_detected_gesture and labels[index] != "enter"and labels[index] != "space":
                    current_pattern.append(labels[index])
                    last_detected_gesture = labels[index]
                    last_detection_time = time.time()


            if labels[index] != "enter" and labels[index] != "space":
                cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                            (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                            (x + w + offset, y + h + offset), (255, 0, 255), 4)

                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)

    # Tombol 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Menampilkan dan menyimpan pola yang telah direkam
print("Recorded Pattern:", current_pattern)

morse_code_mapping = {
    'dash': '-',
    'new': ' ',
    'dit': '.',
    'next': '',
    'space': '/',
}

# Ubah Recorded Pattern menjadi Morse Code
morse_code_pattern = [morse_code_mapping[label] for label in current_pattern]

# Gabungkan Morse Code menjadi satu string
morse_code_string = ''.join(morse_code_pattern)

# Output hasil konversi Morse Code
print("Morse Code Pattern:", morse_code_string)

morse_to_char_mapping = {
    '.-': 'A',
    '-...': 'B',
    '-.-.': 'C',
    '-..': 'D',
    '.': 'E',
    '..-.': 'F',
    '--.': 'G',
    '....': 'H',
    '..': 'I',
    '.---': 'J',
    '-.-': 'K',
    '.-..': 'L',
    '--': 'M',
    '-.': 'N',
    '---': 'O',
    '.--.': 'P',
    '--.-': 'Q',
    '.-.': 'R',
    '...': 'S',
    '-': 'T',
    '..-': 'U',
    '...-': 'V',
    '.--': 'W',
    '-..-': 'X',
    '-.--': 'Y',
    '--..': 'Z',
    '/': ' ',  # Spasi
}

# Memisahkan Morse Code menjadi karakter dengan spasi sebagai delimiter
morse_code_characters = morse_code_string.split(' ')

# Mengonversi setiap karakter Morse Code ke karakter alfabet
decoded_characters = [morse_to_char_mapping[char] if char in morse_to_char_mapping else char for char in morse_code_characters]

# Menggabungkan karakter-karakter untuk membentuk string
decoded_string = ''.join(decoded_characters)

# Output hasil konversi Decoded String
print("Decoded String:", decoded_string)

def text_to_speech(text):
    # Simpan teks ke file sementara
    tts = gTTS(text=text, lang='id')
    tts.save("temp.mp3")

    # Putar file audio menggunakan pemutar default
    os.system("start temp.mp3")

text_to_speech(decoded_string)

cap.release()
cv2.destroyAllWindows()
