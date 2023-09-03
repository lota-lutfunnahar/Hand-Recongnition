import cv2
import numpy as np

image = cv2.imread('IMG-20230830-WA0009.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (15, 15), 0)
_, thresholded = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    if cv2.contourArea(contour) > 5000:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

cv2.imshow('Hand Recognition', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
