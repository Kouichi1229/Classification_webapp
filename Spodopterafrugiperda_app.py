from flask import *
import os
from werkzeug.utils import secure_filename
#from keras.models import load_model
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image


import tensorflow as tf
tf.config.experimental.list_physical_devices('GPU')

app = Flask(__name__)

# Classes of trafic signs
classes = { 0:'Spodoptera frugiperda(秋行軍蟲)',
            1:'Spodoptera litura(斜紋夜盜)',
            2:'Helicoverpa armigera(番茄夜蛾)',
            3:'Spodoptera exigua(甜菜夜蛾)'

      }

def image_processingVGG19(img):
    model = load_model('./model/VGG19_BUG.h5')
    data=[]
    image = Image.open(img)
    image = image.resize((224,224))
    data.append(np.array(image))
    X_test=np.array(data)
    Y_pred = model.predict_classes(X_test)
    return Y_pred

def image_processingInceptionV3(img):
    model = load_model('./model/InceptionV3_bug.h5')
    data=[]
    image = Image.open(img)
    image = image.resize((229,229))
    data.append(np.array(image))
    X_test=np.array(data)
    Y_pred = model.predict_classes(X_test)
    return Y_pred

def image_processingDenseNet201(img):
    model = load_model('./model/DenseNet201_bug.h5')
    data=[]
    image = Image.open(img)
    image = image.resize((224,224))
    data.append(np.array(image))
    X_test=np.array(data)
    Y_pred = model.predict_classes(X_test)
    return Y_pred




@app.route('/')
def index():
    return render_template('InceptionV3.html')

@app.route('/VGG19')
def indexVGG19():
    return render_template('VGG19.html')    

@app.route('/DenseNet201')
def indexDenseNet201():
    return render_template('DenseNet201.html')  





@app.route('/predictVGG19', methods=['GET', 'POST'])
def uploadVGG19():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make prediction
        result = image_processingVGG19(file_path)
        s = [str(i) for i in result]
        a = int("".join(s))
        result = "Predicted is: " +classes[a]
        os.remove(file_path)
        return result
    return None

@app.route('/predictInceptionV3', methods=['GET', 'POST'])
def uploadInceptionV3():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make prediction
        result = image_processingInceptionV3(file_path)
        s = [str(i) for i in result]
        a = int("".join(s))
        result = "Predicted is: " +classes[a]
        os.remove(file_path)
        return result
    return None

@app.route('/predictDenseNet201', methods=['GET', 'POST'])
def uploadDenseNet201():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make prediction
        result = image_processingDenseNet201(file_path)
        s = [str(i) for i in result]
        a = int("".join(s))
        result = "Predicted is: " +classes[a]
        os.remove(file_path)
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)