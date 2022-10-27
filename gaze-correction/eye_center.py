'''import cv2
left_eye_point_x = 0
right_eye_point_x = 0
face_cascade = cv2.CascadeClassifier('face.xml')
eye_cascade = cv2.CascadeClassifier('eye.xml')

cap = cv2.VideoCapture(0)  # sets up webcam

while 1:  # capture frame, converts to greyscale, looks for faces
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:  # draws box around face
        print(x,y,w,h)
        cv2.rectangle(img, (x, y), (x + int(w / 2), y + int(h / 2)), (255, 0, 0), 2)  #
        roi_gray = gray[y:y + int(h / 2), x:x + int(w / 2)]  
        roi_color = img[y:y + h, x:x + w]  
        eyes = eye_cascade.detectMultiScale(roi_gray)  # looks for eyes
        for (ex, ey, ew, eh) in eyes:  # draws boxes around eyes
           if (ex+ey)/2 < (x+y)/2:
                left_eye_point_x = (ex+ey)/2
           elif (ex+ey)/2 > (x+y)/2:
                right_eye_point_x = (ex+ey)/2
           cv2.line(img, (int(left_eye_point_x), int(right_eye_point_x)), (int(left_eye_point_x), int(right_eye_point_x)), (0, 0, 0), 10)

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

import cv2
import numpy as np
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret is False:
        break
    roi = frame[269: 795, 537: 1416]
    rows, cols, _ = roi.shape
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
    _, threshold = cv2.threshold(gray_roi, 3, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        #cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
        break
    cv2.imshow("Threshold", threshold)
    cv2.imshow("gray roi", gray_roi)
    cv2.imshow("Roi", roi)
    key = cv2.waitKey(30)
    if key == 27:
        break
cv2.destroyAllWindows()
'''
'''
import cv2
import numpy as np

# Read image.
img = cv2.imread('eye.png', cv2.IMREAD_COLOR)

# Convert to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur using 3 * 3 kernel.
gray_blurred = cv2.blur(gray, (3, 3))

# Apply Hough transform on the blurred image.
detected_circles = cv2.HoughCircles(gray_blurred,
				cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
			param2 = 30, minRadius = 1, maxRadius = 40)

# Draw circles that are detected.
if detected_circles is not None:

	# Convert the circle parameters a, b and r to integers.
	detected_circles = np.uint16(np.around(detected_circles))

	for pt in detected_circles[0, :]:
		a, b, r = pt[0], pt[1], pt[2]

		# Draw the circumference of the circle.
		cv2.circle(img, (a, b), r, (0, 255, 0), 2)

		# Draw a small circle (of radius 1) to show the center.
		cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
		cv2.imshow("Detected Circle", img)
		cv2.waitKey(0)
'''
'''
    Copyright 2017 by Satya Mallick ( Big Vision LLC )
    http://www.learnopencv.com
'''
'''
import cv2
import numpy as np


def fillHoles(mask):
    
    maskFloodfill = mask.copy()
    h, w = maskFloodfill.shape[:2]
    maskTemp = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(maskFloodfill, maskTemp, (0, 0), 255)
    mask2 = cv2.bitwise_not(maskFloodfill)
    return mask2 | mask

if __name__ == '__main__' :

    # Read image
    img = cv2.imread("red_eyes.jpg", cv2.IMREAD_COLOR)
    
    # Output image
    imgOut = img.copy()
    
    # Load HAAR cascade
    eyesCascade = cv2.CascadeClassifier("eye.xml")
    
    # Detect eyes
    eyes = eyesCascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(100, 100))
    
    # For every detected eye
    for (x, y, w, h) in eyes:

        # Extract eye from the image
        eye = img[y:y+h, x:x+w]

        # Split eye image into 3 channels
        b = eye[:, :, 0]
        g = eye[:, :, 1]
        r = eye[:, :, 2]
        
        # Add the green and blue channels.
        bg = cv2.add(b, g)

        # Simple red eye detector.
        mask = (r<30 and b<30 and g<30)
        
        # Convert the mask to uint8 format.
        mask = mask.astype(np.uint8)*255

        # Clean mask -- 1) File holes 2) Dilate (expand) mask.
        mask = fillHoles(mask)
        mask = cv2.dilate(mask, None, anchor=(-1, -1), iterations=3, borderType=1, borderValue=1)

        # Calculate the mean channel by averaging
        # the green and blue channels
        mean = bg / 2
        mask = mask.astype(np.bool)[:, :, np.newaxis]
        mean = mean[:, :, np.newaxis]

        # Copy the eye from the original image.
        eyeOut = eye.copy()

        # Copy the mean image to the output image.
        #np.copyto(eyeOut, mean, where=mask)
        eyeOut = np.where(mask, mean, eyeOut)

        # Copy the fixed eye to the output image.
        imgOut[y:y+h, x:x+w, :] = eyeOut

    # Display Result
    cv2.imshow('Red Eyes', img)
    cv2.imshow('Red Eyes Removed', imgOut)
    cv2.waitKey(0)
'''
import cv2
img = cv2.imread('eye.png',0)
cv2.imshow("opimg",img)
orb = cv2.ORB_create(200)
keypoint, des = orb.detectAndCompute(img, None)
img_final = cv2.drawKeypoints(img, keypoint, None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
cv2.imshow("op",img_final)
cv2.waitKey(0)