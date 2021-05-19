from objects.AslPredictor import AslPredictor
from config.params import get_model_params, get_dataset_params, get_callback_params

if __name__ == '__main__':
    predictor = AslPredictor(get_model_params(), get_dataset_params(), get_callback_params())
    predictor.run()

