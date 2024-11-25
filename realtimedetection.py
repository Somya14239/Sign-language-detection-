import cv2
import numpy as np
from keras.models import load_model

model = load_model("signlanguagedetectionmodel48x48.h5")

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

cap = cv2.VideoCapture(0)
label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'M', 'N', 'O', 'S', 'T', 'U', 'V', 'W', 'blank']

while True:
    _, frame = cap.read()
    cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
    crop_frame = frame[40:300, 0:300]
    crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
    crop_frame = cv2.resize(crop_frame, (48, 48))
    crop_frame = extract_features(crop_frame)
    pred = model.predict(crop_frame)
    prediction_label = label[np.argmax(pred)]
    cv2.putText(frame, '%s' % (prediction_label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
