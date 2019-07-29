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

# For string randomization
import random, string

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

def take_photo():
    print("Taking photo..")
    # Open new stream
    vcap = cv2.VideoCapture(0)

    # Wait a bit so camera "is ready"
    #time.sleep(2)
    vcap.set(3,1280)
    vcap.set(4,1024)
    time.sleep(2)
    vcap.set(15, -8.0)

    # Get the image
    retval, image = vcap.read()

    # We're done here
    vcap.release()

    # Define jpeg quality
    params = [int(cv2.IMWRITE_JPEG_QUALITY), 100]

    # Brighten image
    #image = change_brightness(image, 1.15, 30)

    file_path = "./photos/" + randomword(12) + ".jpg"

    cv2.imwrite(file_path, image, params)
    return file_path
    # Crop img
    #img = cv2.imread("orig.jpg")
    #crop_img = img[50:900, 30:800]
    #cv2.imwrite("crop.jpg", crop_img, params)


def change_brightness(img, alpha, beta):
    return cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype),0, beta)

def predict(img_path):
    # Load pretrained model
    learn = load_learner('.', 'trained_model.pkl')
    print("Running prediction..")
    # Load image
    img = open_image(img_path)
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
        img_path = take_photo()
        #time.sleep(2)
        prediction = predict(img_path)
        prediction_result = str(prediction[0])
        conf_no = float(prediction[1][0].item())
        conf_wtf = float(prediction[1][1].item())
        conf_yes = float(prediction[1][2].item())
        return {'coffee': prediction_result, 'confidence_yes': conf_yes, 'confidence_no': conf_no, 'confidence_wtf': conf_wtf}
        #return {'coffee': prediction_result, 'prediction': str(prediction[1])}

api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8000', debug=True)
