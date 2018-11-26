# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
output = image.copy()

# load the trained convolutional neural network and the label
# binarizer
print("[INFO] loading network...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())


nms = np.zeros([50,75])

thumb = cv2.resize(image, (400,300))
#orig ranges were 100,200 / 133,266
'''
for i in range(125,175):
    for j in range(150,225):
        y = i*10
        x = j*10
        subImg = image[y:y+100, x:x+100]

        # pre-process the image for classification
        img = cv2.resize(subImg, (100, 100))
        img = img.astype("float") / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)

        # classify the input image
        #print("[INFO] classifying image...")
        proba = model.predict(img)[0]
        idx = np.argmax(proba)
        label = lb.classes_[idx]

        if(label == 'PositiveSamplesCorn' and max(proba) > .85):
            print([y,x])
            print([i-125, j-150])
            nms[i-125][j-150] = max(proba)
            xc = int(x)
            yc = int(y)
            cv2.rectangle(image, (x,y), (x+25, y+25), (255,0,0), -1)
'''
print(nms)
np.savetxt('nms', nms)

#orig = 900:2100, 1233:2766
cv2.imshow("corn",image[1150:1875, 1400:2350])
cv2.waitKey(0)


'''
# we'll mark our prediction as "correct" of the input image filename
# contains the predicted label text (obviously this makes the
# assumption that you have named your testing image files this way)
filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
correct = "correct" if filename.rfind(label) != -1 else "incorrect"

# build the label and draw the label on the image
label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
output = imutils.resize(output, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# show the output image
print("[INFO] {}".format(label))
cv2.imshow("Output", output)
cv2.waitKey(0)
'''
