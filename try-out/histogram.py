#using matplotlib
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
#read as gray
img = cv.imread('try-out/cat.jpeg',0)
plt.hist(img.ravel(),256,[0,256]); 
plt.show()

#merging for all color channels in plt
img = cv.imread('try-out/cat.jpeg')
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

# without opencv

# function to obtain histogram of an image
def hist_plot(img):
	count =[]
	r = []
	for k in range(0, 256):
		r.append(k)
		count1 = 0
		for i in range(m):
			for j in range(n):
				if img[i, j]== k:
					count1+= 1
		count.append(count1)
	return (r, count)
img = cv.imread('try-out/cat.jpeg',0)
m, n = img.shape[:2]
r1, count1 = hist_plot(img)
plt.stem(r1, count1)
plt.xlabel('intensity value(0-255)')
plt.ylabel('number of pixels')
plt.title('Histogram of the gray image')

# Transformation to obtain stretching
constant = (255-0)/(img.max()-img.min())
img_stretch = img * constant
r, count = hist_plot(img_stretch)

# plotting the histogram
plt.stem(r, count)
plt.xlabel('intensity value')
plt.ylabel('number of pixels')
plt.title('Histogram of the stretched image')

# Storing stretched Image
# cv.imwrite('Stretched Image 4.png', img_stretch)
plt.show()
#using opencv
# create a mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
masked_img = cv.bitwise_and(img,img,mask = mask)
# Calculate histogram with mask and without mask
# Check third argument for mask
hist_full = cv.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv.calcHist([img],[0],mask,[256],[0,256])
plt.subplot(221), plt.imshow(img)
plt.subplot(222), plt.imshow(mask)
plt.subplot(223), plt.imshow(masked_img)
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0,255])
plt.show()
#using numpy
plt.subplot(1,2,1)
plt.imshow(img,cmap='gray')
plt.title('image')
plt.xticks([])
plt.yticks([])

plt.subplot(1,2,2)
hist,bin = np.histogram(img.ravel(),256,[0,255])
plt.xlim([0,255])
plt.plot(hist)
plt.title('histogram')

plt.show()