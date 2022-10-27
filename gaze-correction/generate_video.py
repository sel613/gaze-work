import time,cv2
t_end = time.time() + 60*15
image = cv2.imread('/home/digital/selvapriyanka/AI-ML/gaze-correction/Elon_Musk.jpg')
size = (image.shape[1],image.shape[0])
image_array=[]
result = cv2.VideoWriter('elon.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),15, size)
i=0
while i<500:
  print(i)
  i+=1
  image_array.append(image)

for i in range(len(image_array)):
  result.write(image_array[i])
result.release()

 