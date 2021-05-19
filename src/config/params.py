def get_model_params():
    return {
        'transfer_learning_model_name': 'imagenet',
        'lr': 1e-4,
        'momentum': 0.9,
        'loss_function': 'categorical_crossentropy',
        'metrics': ['acc'],
        'first_trainable_layer': 249,

        # TRAINING PARAMS
        'steps_per_epoch': 200,
        'validation_steps': 50,
        'epochs': 50,
    }


def get_dataset_params():
    return {
        'training_dir': '../data',
        'target_size': (200, 200),
        'shuffle': True,
        'seed': 13,
        'class_mode': 'categorical',
        'batch_size': 64,
    }


def get_callback_params():
    return {
        'loss_threshold': 0.2,
        'accuracy_threshold': 0.95,
    }


def get_image_data_generator_params():
    return {
        'samplewise_center': True,
        'samplewise_std_normalization': True,
        'brightness_range': [0.8, 1.0],
        'zoom_range': [1.0, 1.2],
        'validation_split':  0.1
    }
