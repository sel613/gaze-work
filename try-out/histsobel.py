# reduce no the global variables
# u should have a main function where u r reading image from path
# resize the image to 512x512
# then pass it to pyCalcHist function (no extra args just image will work)
# pyCalcHist should return me a 1D list, where list values are the histogram of the image
# then pass the image to py3x3_sobel (no extra args just image will work)
# this should return me output image after applying 3x3 sobel filter
# implement 2d 3x3 filter and dont use two 1d filter, you should have sliding window inside py3x3_sobel
import cv2
import numpy as np
def pyCalcHist(img):
    k=0
    hist=np.zeros((256),dtype=int)
    while k<256:
        freq=0
        for i in range(512):
            for j in range(512):
                if img[i][j]==k:
                    freq+=1
        hist[k]=freq
        print(k,freq)
        k+=1
    return hist

def py3_3_sobel(img):
    print(img.shape)
    stepSize=1
    op = np.zeros(img.shape)
    img=np.pad(img, pad_width=1)
    # print(img.shape)
    filter = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
    for y in range(0, img.shape[0]-3, stepSize):
        for x in range(0, img.shape[1]-3, stepSize):
            p=img[y:y +filter.shape[1], x:x + filter.shape[0]]
            p = p.flatten() *filter.flatten()
            sum = p.sum()
            if sum<0:
                sum=0
            op[y - filter.shape[1]][x - filter.shape[0]] = sum
    return op


def main(img_path):
    img = cv2.imread(img_path)
    # print(img)
    cv2.imshow("img",img)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (512,512),interpolation=cv2.INTER_AREA)
    print("*********")
    hist=pyCalcHist(img)
    print(hist)
    op_img=py3_3_sobel(img)
    print(op_img.shape)
    cv2.imshow("sobel detector",op_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    image_path="image.jpg"
    main(image_path)
