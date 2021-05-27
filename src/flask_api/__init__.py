from flask import Flask, render_template, request
from config.flask_config import Config
import os

from objects.AslPredictor import AslPredictor
from config.params import get_model_params, get_dataset_params, get_callback_params, clean_dir

# Create Flask application
flask_app = Flask(__name__)
flask_app.config.from_object(Config)


# ______________________________________________________________________________________________________________________
@flask_app.route('/')
@flask_app.route('/home')
def home():
    return render_template("index.html")


# ______________________________________________________________________________________________________________________
@flask_app.route("/prediction", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        post_data = request.json

        model_name = 'default_model'
        if 'model' in post_data.keys():
            model_name = post_data['model_name']

        predictor = AslPredictor(get_model_params(), get_dataset_params(), get_callback_params())
        predictor.load_model(model_name)

        print(image_file.filename)
        if image_file:
            prediction_dir = flask_app.config['UPLOAD_PRED_FOLDER']

            # clean prediction directory before prediction
            clean_dir(prediction_dir)

            image_location = os.path.join(
                prediction_dir,
                image_file.filename
            )

            image_file.save(image_location)

            pred = predictor.predict(prediction_dir)[0]

            return render_template("prediction.html", prediction=pred, image_loc=image_file.filename)

    return render_template("prediction.html", prediction=0, image_loc=None)


# ______________________________________________________________________________________________________________________
@flask_app.route("/training", methods=["GET"])
def training():
    predictor = AslPredictor(get_model_params(), get_dataset_params(), get_callback_params())
    predictor.run()

    return render_template("index.html")


# ======================================================================================================================
def create_app(config_class=Config):
    
    # random database creation
    with flask_app.app_context():

        return flask_app

