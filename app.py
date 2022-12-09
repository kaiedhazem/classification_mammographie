
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf
keras = tf.keras
from keras.utils import load_img, img_to_array
import numpy as np
from PIL import Image
import os
import io

app = Flask(__name__)

model = tf.keras.models.load_model('C:/Users/asus/Desktop/IAproject/model')

def predict_class(image):

	img = load_img(image, target_size=(128, 128), color_mode='grayscale')
	x = img_to_array(img)
	x = np.expand_dims(x, axis=0)
	images = np.vstack([x])
	images /= 255.0
	prediction = model.predict(images)

	return prediction

app.config["UPLOAD_FOLDER"] = "static/"

@app.route('/')
def upload_file():
    return render_template('index.html')


@app.route('/display', methods = ['GET', 'POST'])
def save_file():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)

        f.save(app.config['UPLOAD_FOLDER'] + filename)
        image=Image.open("C:/Users/asus/Desktop/IAproject/static/"+ filename)
        buffer = io.BytesIO() # create file in memory
        image.save(buffer, 'jpeg') # save in file in memory - it has to be jpeg, not jpg
        buffer.seek(0) # move to the beginning of file
        bg_image = buffer # use it without open()
        pred = predict_class(bg_image)
        class_names = ['a du cancer benine', 'a du cancer', 'est normale']
        result = class_names[np.argmax(pred)]
        output = 'La Personne ' + result
        os.remove("C:/Users/asus/Desktop/IAproject/static/"+ filename)
    return render_template('index.html', content=output) 

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug = True,use_reloader=False)