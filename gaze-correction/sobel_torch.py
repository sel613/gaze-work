import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import math
def conv2d(img,filter):
    stepSize=1
    # op=torch.zeros(img.shape[0],img.shape[1])
    c=[]
    for y in range(0, img.shape[0]-3, stepSize):
        for x in range(0, img.shape[1]-3, stepSize):
            p=img[y:y +filter.shape[1], x:x + filter.shape[0]]
            c = torch.cat([p], dim=1)
            p=torch.matmul(p,filter)
            # p_y=img[y:y +filter_y.shape[1], x:x + filter_y.shape[0]]
            # p_y = p_y.flatten() *filter_y.flatten()
            # sum_y = p_y.sum()
            # sum=math.sqrt(sum_x**2 + sum_y**2)
    print(c)
    return p
def py3_3_sobel(img):
    img=np.pad(img, pad_width=1)
    convert_tensor = transforms.ToTensor()
    img=convert_tensor(img)
    filter_x=torch.Tensor([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
    filter_y=torch.Tensor([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
    filter_x_op=conv2d(img,filter_x)
    filter_y_op=conv2d(img,filter_y)
    sum=math.sqrt(filter_x_op**2 + filter_y_op**2)
    return sum
    # op = np.zeros(img.shape)
    # img=np.pad(img, pad_width=1)
    # # input = tf.to_tensor(img);
    # convert_tensor = transforms.ToTensor()
    # img=convert_tensor(img)
    # # input = input.unsqueeze_(0);
    # # print(img.shape)
    # # Conv2d(in_channels, out_channels, kernel_size=(n, n), stride, padding, bias)
    # filter_x = torch.from_numpy(np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]]))
    # filter_y = torch.from_numpy(np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]]))
    # p_x=nn.Conv2d
    # return op
def main(img_path):
    img = cv2.imread(img_path)
    # print(img)
    cv2.imshow("img",img)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (512,512),interpolation=cv2.INTER_AREA)
    print("*********")
    op_img=py3_3_sobel(img)
    print(op_img.shape)
    cv2.imshow("sobel detector",op_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__=="__main__":
    image_path="image.jpeg"
    main(image_path)

'''
# Here we define the matrices associated with the Sobel filter
Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
[rows, columns] = np.shape(grayscale_image)  # we need to know the shape of the input grayscale image
sobel_filtered_image = np.zeros(shape=(rows, columns))  # initialization of the output image array (all elements are 0)

# Now we "sweep" the image in both x and y directions and compute the output
for i in range(rows - 2):
    for j in range(columns - 2):
        gx = np.sum(np.multiply(Gx, grayscale_image[i:i + 3, j:j + 3]))  # x direction
        gy = np.sum(np.multiply(Gy, grayscale_image[i:i + 3, j:j + 3]))  # y direction
        sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  # calculate the "hypotenuse"

'''