import os


class Config:
    
    # Create dummy secrey key so we can use sessions
    SECRET_KEY = '123456790'

    DEBUG = True
    TESTING = False
    ENV ='development'
    FLASK_API_ROOT_PATH = os.path.join(os.path.realpath(os.path.dirname(__file__)))
    UPLOAD_PRED_FOLDER = os.path.join( os.path.realpath( os.path.dirname(FLASK_API_ROOT_PATH) ) ,'flask_api', 'static',  'upload_pred_images' )
    