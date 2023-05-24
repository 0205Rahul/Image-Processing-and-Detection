import numpy as np
import sys
import cv2
import io,os
import tkinter as tk
from tkinter import filedialog
import ipywidgets as widgets
from PIL import Image
from IPython.display import display,clear_output

img_1= cv2.imread('q1.jpg')
img_2 = cv2.imread('q2.jpg')


cv2.imshow("Image 1", img_1)
cv2.imshow("Image 2", img_2)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_1.shape

img_2.shape

diff = cv2.absdiff(img_1, img_2) #find pixel by pixel difference wrt to both images

diff

gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

gray

_, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
thresh = cv2.dilate(thresh, kernel, iterations=2)
thresh = cv2.erode(thresh, kernel, iterations=1)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

thresh_limit = 200

segmented_diff = img_2.copy()

i = 0
for ct in (contours):
    area_mark = cv2.contourArea(ct)
    if area_mark > thresh_limit:
        (x, y, w, h) = cv2.boundingRect(ct)
        cv2.rectangle(segmented_diff, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(segmented_diff, f'Diff {i}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        i += 1

cv2.imshow("Difference", segmented_diff)
cv2.waitKey(0)
cv2.destroyAllWindows()

save_dir = "image_data"
os.makedirs(save_dir, exist_ok=True)

# Save the captured/uploaded images
cv2.imwrite(os.path.join(save_dir, "realim1.jpg"), img_1)
cv2.imwrite(os.path.join(save_dir, "realim2.jpg"), img_2)
cv2.imwrite(os.path.join(save_dir, "difference.jpg"), segmented_diff)

