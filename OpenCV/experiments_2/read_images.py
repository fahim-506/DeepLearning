import cv2

img1 = cv2.imread('../baby.png')
cv2.imshow('The original image',img1)
cv2.waitKey(0)

cv2.imwrite('saved_img.jpg',img1)
