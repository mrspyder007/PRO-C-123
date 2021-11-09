import cv2
import pandas as pd
import numpy as np
import csv
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time
import seaborn as sb
import matplotlib.pyplot as plt
X=np.load('image.npz')['arr_0']
y=pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses = len(classes)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=8,train_size=3500,test_size=500)
X_train_scaled=X_train/255.0
X_test_scaled=X_test/255.0
clf=LogisticRegression(solver='saga',multi_class='multinomial').fit(X_train_scaled,X_test_scaled)

y_pred=clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("The accuracy is :- ",accuracy)
cap = cv2.VideoCapture(0)
print("camera started")
while(True):
    Capture=frame-frame
    try:
      ret, frame = cap.read()
      print("Inside the loop")
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 #Drawing a box in the center of the video
      height, width = gray.shape
      upper_left = (int(width / 2 - 56), int(height / 2 - 56))
      bottom_right = (int(width / 2 + 56), int(height / 2 + 56))
      cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

      #roi = Region Of Interest
      roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
      im_pil = Image.fromarray(roi)

    # # convert to grayscale image - 'L' format means each pixel is 
    # # represented by a single value from 0 to 255
      image_bw = im_pil.convert('L')
      image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
    # #invert the image
      image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
      pixel_filter = 20
    # #converting to scalar quantity
      min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
      #using clip to limit the values between 0,255
      image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
      max_pixel = np.max(image_bw_resized_inverted)
      #converting into an array
      image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    # #creating a test sample and making a prediction
      test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
      test_pred = clf.predict(test_sample)
      print("Predicted class is: ", test_pred)

#     # Display the resulting frame
      cv2.imshow('frame',gray)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    except Exception as e:
      pass

# # When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
