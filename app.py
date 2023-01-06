import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
from flask import Flask, jsonify, request
from markupsafe import escape
import io


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load_model():
    global model
    model = keras.models.load_model('final_model.h5')
    print('model loaded')
    
def prepare_image(image):
    # if the image mode is not RGB, convert it
    # if image.mode != "RGB":
    #     image = image.convert("RGB")

    # resize the input image and preprocess it
    # image = image.resize(target)
    image = img_to_array(image)
    image = tf.keras.utils.normalize(image, axis = 1)
    image = np.array(image).reshape(-1,28, 28,1)

    # return the processed image
    return image

@app.route("/", methods=["POST", "GET"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        print('debug1')
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            print('debug')
            # preprocess the image and prepare it for classification
            image = prepare_image(image)
            # print('image:',image)

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            print(preds)
            print(np.argmax(preds,axis=1))
            response = np.array_str(np.argmax(preds,axis=1))
            # response = np.array_str(np.argmax(preds,axis=1))
            
            # results = preds
            # results = imagenet_utils.decode_predictions(preds)
            # data["predictions"] = []

            # # loop over the results and add them to the list of
            # # returned predictions
            # for (imagenetID, label, prob) in results[0]:
            #     r = {"label": label, "probability": float(prob)}
            #     data["predictions"].append(r)

            # # indicate that the request was a success
            # data["success"] = True

    # return the data dictionary as a JSON response
    # return flask.jsonify(data)
    return response

if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_model()
	app.run()