import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from math import sqrt


class sobel_edge_detector:
    def __init__(self, path= None, img= None):
        if path != None:
            self.image = mpimg.imread(path)
        else:
            self.image = img
        self.orig_img = self.image
        self.vertical_grad_filter = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
        self.horizontal_grad_filter = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])

    def cvt2gray(self):
        self.image = np.dot(self.image, [1, 1, 1])//3
        self.image = self.image/255

    def detect_edges(self):
        self.cvt2gray()
        kernel_width = self.vertical_grad_filter.shape[0]//2
        grad_ = np.zeros(self.image.shape)

        self.image = np.pad(self.image, pad_width= ([kernel_width, ], [kernel_width, ]), 
        mode= 'constant', constant_values= (0, 0))
        for i in range(kernel_width, self.image.shape[0] - kernel_width):
            for j in range(kernel_width, self.image.shape[1] - kernel_width):
                x = self.image[i - kernel_width: i + kernel_width + 1, j - kernel_width: j + kernel_width + 1]
                x = x.flatten() * self.vertical_grad_filter.flatten()
                sum_x = x.sum()
                y = self.image[i - kernel_width: i + kernel_width + 1, j - kernel_width: j + kernel_width + 1]
                y = y.flatten() * self.horizontal_grad_filter.flatten()
                sum_y = y.sum()
                grad_[i - kernel_width][j - kernel_width] = sqrt(sum_x**2 + sum_y**2)
        self.image = grad_
        return self.image
    def show_image(self, orig = 0):
        if orig == 1:
            plt.imshow(self.orig_img)
            plt.show()
        if orig == 0:
            for i in range(len(self.image)):
                for j in range(len(self.image[0])):
                    self.image[i][j] = 1 - self.image[i][j] 
            plt.imshow(self.image, cmap= 'gray')
            plt.show()

if __name__ == "__main__":
    img = sobel_edge_detector("edge-detections/cat.jpeg")
    img.detect_edges()
    img.show_image()
    
