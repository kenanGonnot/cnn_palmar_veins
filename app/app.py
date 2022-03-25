import numpy as np
import os
from PIL.Image import Image
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from load import init, model_predict
from werkzeug.utils import secure_filename

app = Flask(__name__)

list_users = ["kenan", "faycal", "lorenzo"]

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# set paths to upload folder
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['IMAGE_UPLOADS'] = os.path.join(APP_ROOT, 'static')

model_architecture_path = "deployed_model/model_identification_500users_1Layer.json"
model_weights_path = "deployed_model/model_identification_500users_1Layer.h5"

model = init(model_architecture_path, model_weights_path)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/identification', methods=["POST", "GET"])
def identification():
    if request.method == "POST":
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        prediction = model_predict(file_path, model)
        pred_class = np.argmax(prediction) + 1
        result = str(pred_class)
        return result
    return render_template('identification.html')


# def identification():
#     # Check if a valid image file was uploaded
#     # if 'file' not in request.files:
#     #     flash('No file part')
#     #     return redirect(request.url)
#     if request.method == "POST":
#         f = request.files['file']
#         # if f == '':
#             # flash('No file selected for uploading')
#             # return redirect(request.url)
#         if f and allowed_file(f.filename):
#             basepath = os.path.dirname(__file__)
#             print("basepath: ", basepath)
#             # file_path = os.path.join(app.config["IMAGE_UPLOADS"], f.filename)
#             file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
#             f.save(file_path)
#             # The image file seems valid! Detect the identity of the palm veins and return the result.
#             preds = model_predict(file_path, model)
#             pred_class = list(preds.argmax(axis=-1))
#             pred_prob = list(preds[0][pred_class[0]])
#             print("preds: ", preds)
#             print("pred_class: ", pred_class)
#             print("pred_prob: ", pred_prob)
#             # return render_template('identification.html',
#             #                        result=pred_class[0],
#             #                        prob=pred_prob[0],
#             #                        img_path=file_path)
#             return render_template('identification.html', uploaded_image=f.filename)
#         else:
#             flash('Allowed file types are png, jpg, jpeg')
#             return redirect(request.url)
#
#         # preds = model_predict(file_path, model)
#         # pred_class = np.argmax(preds, axis=1)
#
#         return render_template('identification.html')
#     else:
#         return render_template('identification.html')


@app.route('/identification/<user>', methods=["POST", "GET"])
def identification_user(user=''):
    return send_from_directory(app.config["IMAGE_UPLOADS"], user)


@app.route('/verification', methods=["POST", "GET"])
def verification():
    if request.method == "POST":
        user = request.form["username"]
        for usr in list_users:
            if user == usr:
                return redirect(url_for('connected', usr=user))
        return redirect(url_for('error', error="Username not found"))
    else:
        return render_template('verification.html')


@app.route('/connected/<usr>')
def connected(usr):
    return render_template('connected.html', usr=usr)


@app.route('/error/<error>')
def error(error):
    return render_template('error.html', error=error)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
