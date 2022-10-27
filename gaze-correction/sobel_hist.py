import math
import cv2
import numpy as np
def pyCalcHist(img):
    k=0
    hist=np.zeros((256),dtype=int)
    for i in range(512):
        for j in range(512):
            hist[img[i][j]]+=1
    return hist
def py3_3_sobel(img):
    print(img.shape)
    stepSize=1
    op = np.zeros(img.shape)
    img=np.pad(img, pad_width=1)
    # print(img.shape)
    filter_x = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
    filter_y = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
    for y in range(0, img.shape[0]-3, stepSize):
        for x in range(0, img.shape[1]-3, stepSize):
            p_x=img[y:y +filter_x.shape[1], x:x + filter_x.shape[0]]
            p_x = p_x.flatten() *filter_x.flatten()
            sum_x = p_x.sum()
            p_y=img[y:y +filter_y.shape[1], x:x + filter_y.shape[0]]
            p_y = p_y.flatten() *filter_y.flatten()
            sum_y = p_y.sum()
            sum=math.sqrt(sum_x**2 + sum_y**2)
            op[y - filter_x.shape[1]][x - filter_x.shape[0]] = sum
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
    image_path="image.jpeg"
    main(image_path)