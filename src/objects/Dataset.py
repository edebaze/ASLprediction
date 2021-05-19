from keras.preprocessing.image import ImageDataGenerator
from config.params import get_image_data_generator_params


class Dataset:
    def __init__(self, dataset_params):
        self.training_dir = dataset_params['training_dir']
        self.target_size = dataset_params['target_size']
        self.shuffle = dataset_params['shuffle']
        self.seed = dataset_params['seed']
        self.class_mode = dataset_params['class_mode']
        self.batch_size = dataset_params['batch_size']

        self.data_generator = None
        self.train_generator = None
        self.validation_generator = None

        self.initialize()

    # __________________________________________________________________________________________________________________
    def initialize(self):
        self.set_data_generator()
        self.set_train_val_generators()

    # __________________________________________________________________________________________________________________
    def set_data_generator(self):
        self.data_generator = ImageDataGenerator(**get_image_data_generator_params())

    # __________________________________________________________________________________________________________________
    def set_train_val_generators(self):
        self.train_generator = self.data_generator.flow_from_directory(
            self.training_dir,
            target_size=self.target_size,
            shuffle=self.shuffle,
            seed=self.seed,
            class_mode=self.class_mode,
            batch_size=self.batch_size,
            subset="training"
        )

        self.validation_generator = self.data_generator.flow_from_directory(
            self.training_dir,
            target_size=self.target_size,
            shuffle=self.shuffle,
            seed=self.seed,
            class_mode=self.class_mode,
            batch_size=self.batch_size,
            subset="validation"
        )
