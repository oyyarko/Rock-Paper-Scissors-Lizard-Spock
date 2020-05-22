import os
import cv2
import sys

try:
    label_name = sys.argv[1]
    class_samples = int(sys.argv[2])
except:
    print("Argument missing")
    exit(-1)

IMG_SAVE_PATH = 'data'
IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, label_name)

try:
    os.mkdir(IMG_SAVE_PATH)
except FileExistsError:
    pass

try:
    os.mkdir(IMG_CLASS_PATH)
except:
    print("Directory already exists")
    print("\nAll images will be gathered")

cap = cv2.VideoCapture(0)
start = False
count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        continue
    
    if count == class_samples:
        break
    
    cv2.rectangle(frame, (100, 100), (400, 400), (255,255,255), 2)
    
    if start:
        img = frame[100:400, 100:400]
        save_path = os.path.join(IMG_CLASS_PATH, "{}.jpg".format(count+1))
        cv2.imwrite(save_path, img)
        count += 1
    
    font=cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(frame, "total: {}".format(count), (5, 50), font, 0.7, (255,0,0), 2, cv2.LINE_AA)
    cv2.imshow('Data collection', frame)
    
    k = cv2.waitKey(10)
    if k == ord('a'):
        start = not start
    
    if k == ord('q'):
        break


print("\n{} images saved to {}".format(count, IMG_CLASS_PATH))
cap.release()
cv2.destroyAllWindows()
    