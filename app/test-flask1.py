from io import BytesIO

# import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from app.utils import load_pretrained_model

UPLOAD_FOLDER = '../data/data_palm_vein/NIR'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANNELS = 1


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pretrained models
model_architecture_path = '../saved_model/model_identification_500users_40epochs.json'
model_weights_path = '../saved_model/model_identification_500users_40epochs.h5'
model = load_pretrained_model(model_architecture_path, model_weights_path)


@app.route('/', methods=['GET'])
def send_index():
    return send_from_directory('./www', "home.html")


@app.route('/<path:path>', methods=['GET'])
def send_root(path):
    return send_from_directory('./www', path)


@app.route('/api/image', methods=['POST'])
def upload_image():
    # check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No posted image. Should be attribute named image.'})
    file = request.files['image']

    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        return jsonify({'error': 'Empty filename submitted.'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        x = []
        # ImageFile.LOAD_TRUNCATED_IMAGES = False
        # img = Image.open(BytesIO(file.read()))
        # img.load()
        # img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
        # x = image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        # x = preprocess_input(x)
        # x = x[:, :, :, 0:3]
        # pred = model.predict(x)
        # lst = decode_predictions(pred, top=5)
        items = []
        # for itm in lst[0]:
        #     items.append({'name': itm[1], 'prob': float(itm[2])})

        img = cv2.imread(BytesIO(file.read()), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        X = np.expand_dims(img, axis=0)
        pred = model.predict(X)
        items.append({'User id :': np.argmax(pred), 'prob': float(max(pred))})

        response = {'pred': items}
        print(response)
        return jsonify(response)
    else:
        return jsonify({'error': 'File has invalid extension'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
