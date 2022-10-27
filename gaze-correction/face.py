import face_alignment
from skimage import io
import numpy as np
import cv2
import matplotlib.pyplot as plt
cpu=1
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,device= 'cpu' if cpu else 'cuda')

input = io.imread('2022-10-24-135320.jpg')
print(fa)
preds = fa.get_landmarks(255 * input)[0]
# print(preds)
# print(preds.shape)
# preds=np.asarray(preds)
print(preds.shape)
pred=preds[36:48,:]
print(pred)
for i  in range(36,48):
#     for i in range(startpoint, endpoint+1):
#     point = [faceLandmarks.part(i).x, faceLandmarks.part(i).y]
    # points.append(point)
    cv2.circle(input, (int(preds[i][0]),int(preds[i][1])), 1, (0, 0, 255), -1)
    
# for i, (x, y) in enumerate(preds):
# 	    # Draw the circle to mark the keypoint 
#     cv2.circle(input, (int(x),int( y)), 1, (0, 0, 255), -1)
#     cv2.putText(input, str(i),(int(x),int( y)),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 255),1)# # plt.plot(preds[:,0], -preds[:,1], 'ro', markersize=8, alpha = 0.5)
# # for i in range(preds.shape[0]):
#     plt.text(preds[i,0]+1, -preds[i,1], str(i), size=14)
# fig = plt.figure(figsize=(15, 5))
# ax = fig.add_subplot(1, 3, 1)
# ax.imshow(input)
# ax = fig.add_subplot(1, 3, 2)
# ax.scatter(preds[:, 0], -preds[:, 1], alpha=0.8)
# ax = fig.add_subplot(1, 3, 3)
# # img2 = img.copy()
img2 = input.copy()
cv2.imshow('Landmark Detection', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
# for p in preds:
#     img2[p[1]-3:p[1]+3, p[0]-3:p[0]+3, :] = 255
    # note that the values -3 and +3 will make the landmarks
    # overlayed on the image 6 pixels wide; depending on the
    # resolution of the face image, you may want to change
    # this value

# plt.imshow(img2)
# plt.show()
# kp_source = fa.get_landmarks(255 * input)[0]
# print(preds)
# print(kp_source)