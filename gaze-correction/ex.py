import cv2
 
image = cv2.imread("beauty.jpg")
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
eyes = eye_cascade.detectMultiScale(image, scaleFactor = 1.2,
                                    minNeighbors = 4)
 
 
for (x,y,w,h) in eyes:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0, 255, 0),5)
 
cv2.imshow("Eye Detected", image)
cv2.waitKey(0)
cv2.destroyAllWindows()