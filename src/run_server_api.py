from objects.AslPredictor import AslPredictor
from config.params import get_model_params, get_dataset_params, get_callback_params

from flask_api import create_app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', debug=True,port=5550)

