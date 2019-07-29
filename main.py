# For FastAI
from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate

# For openCv
import cv2
import time

# For Flask
from flask import Flask
from flask_restful import Resource, Api

def take_photo():
    print("Taking photo..")
    # Open new stream
    vcap = cv2.VideoCapture(0)

    # Wait a bit so camera "is ready"
    time.sleep(2)

    # Get the image
    retval, image = vcap.read()

    # We're done here
    vcap.release()

    # Define jpeg quality
    params = [int(cv2.IMWRITE_JPEG_QUALITY), 100]

    image = change_brightness(image, 1.15, 30)

    cv2.imwrite("orig.jpg", image, params)

    # Crop img
    #img = cv2.imread("orig.jpg")
    #crop_img = img[50:900, 30:800]
    #cv2.imwrite("crop.jpg", crop_img, params)


def change_brightness(img, alpha, beta):
    return cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype),0, beta)

def predict():
    # Load pretrained model
    learn = load_learner('.', 'trained_model.pkl')
    print("Running prediction..")
    # Load image
    img = open_image('orig.jpg')
    # Run prediction against image
    prediction_class, prediction_idx, outputs = learn.predict(img)
    # Print result and show source image
    print("Is there coffee in the Moccamaster? The answer is:", prediction_class)
    print("Confidence distribution for the image is:", outputs)
    return [prediction_class, outputs]

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        take_photo()
        prediction = predict()
        prediction_result = str(prediction[0])
        conf_no = float(prediction[1][0].item())
        conf_wtf = float(prediction[1][1].item())
        conf_yes = float(prediction[1][2].item())
        return {'coffee': prediction_result, 'confidence_yes': conf_yes, 'confidence_no': conf_no, 'confidence_wtf': conf_wtf}

api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8000', debug=True)