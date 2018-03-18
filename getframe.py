import cv2

vidcap = cv2.VideoCapture('new_video_0.mp4')
count = 0
success = True
while success and count<10:
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  count += 1