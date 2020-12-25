from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
from keras.models import load_model
import cv2
app = Flask(__name__)

model = load_model("HandWrittenModel.h5")

@app.route('/hello')
def hello():
   return render_template('hello.html')


def preprocess_img(filepath , newshape):
    img = cv2.imread(filepath , cv2.IMREAD_GRAYSCALE)
    img = img / 255
    img = cv2.resize(img , newshape)
    img = np.reshape(img , (1 , newshape[0] , newshape[1] , 1))
    return img

def Predict(img):
    labels_prop = model.predict(img)
    label = np.argmax(labels_prop)
    return label

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      img = preprocess_img(f.filename , (28 , 28))
      predict = Predict(img)
      return render_template("final.html" , data = predict) 
		
if __name__ == '__main__':
   app.run(debug = True)