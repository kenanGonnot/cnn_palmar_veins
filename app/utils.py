from keras.models import model_from_json


def load_pretrained_model(model_architecture_path, model_weights_path):
    json_file = open(model_architecture_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_weights_path)
    print("Loaded Model from disk")
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model

