import cv2

vidcap = cv2.VideoCapture('newv.mp4')
count = 0
success = True
frame_time = 9000.2
vidcap.set(cv2.CAP_PROP_POS_MSEC,frame_time)
if success:
  success,image = vidcap.read()
  print('Read a new frame at time %f: '%frame_time, success)
  image = cv2.resize(image, (224,224)) 
  print image.shape
  cv2.imwrite("frame%d.jpg" %frame_time, image)     # save frame as JPEG file
  ############ image is our sample to be fed into the NN
  count += 1