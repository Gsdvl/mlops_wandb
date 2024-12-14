import pandas as pd
import argparse, os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras import optimizers
from wandb.integration.keras import WandbMetricsLogger
from types import SimpleNamespace
from fastai.data.external import Path


import wandb
import params

def get_data():
    processed_data_at = run.use_artifact(f'{params.PROCESSED_DATA_AT}:latest')
    processed_dataset_dir = Path(processed_data_at.download())
    cwd = Path(os.getcwd())
    path = processed_dataset_dir.relative_to(cwd)
    
    return f'{path}/flowers'

def generate_train():
    dataset_dir = 'artifacts/flowers_simples_split:v3/flowers'

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=30,      # Gira a imagem até 30 graus
        width_shift_range=0.2,  # Translação horizontal
        height_shift_range=0.2, # Translação vertical
        shear_range=0.2,        # Cisalhamento
        zoom_range=0.2,         # Zoom
        horizontal_flip=True    # Flip horizontal
    )

    # Gerador para treinamento
    train_generator = datagen.flow_from_directory(
        directory=dataset_dir,
        target_size=(180, 180),
        batch_size=model_configs.batch_size,
        class_mode='categorical',
        subset='training' 
    )

    # Gerador para validação
    validation_generator = datagen.flow_from_directory(
        directory=dataset_dir,
        target_size=(180, 180),
        batch_size=model_configs.batch_size,
        class_mode='categorical',
        subset='validation'
    )
    return train_generator, validation_generator

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(5, activation='softmax') 
    ])

    my_optimizer = optimizers.Adam(learning_rate=model_configs.lr, beta_1=model_configs.beta_1, beta_2=model_configs.beta_2)

    model.compile(optimizer=my_optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    model.summary()
    return model

def train_model(model, train_generator ,validation_generator):
    history = model.fit(
        train_generator,
        epochs=model_configs.epochs,
        validation_data=validation_generator,
        callbacks=[wandb.keras.WandbMetricsLogger(),
                        wandb.keras.WandbModelCheckpoint(filepath='model.keras', save_best_only=True)])
    

model_configs = SimpleNamespace(
    lr=0.001,
    batch_size=32,
    epochs=10,
    beta_1 = 0.9,
    beta_2 = 0.999
)
    
def parse_args():
    argparser = argparse.ArgumentParser(description="Process Hyperparameters for Model Training")
    
    argparser.add_argument('--batch_size', type=int, default=model_configs.batch_size, help='Batch size for training')
    argparser.add_argument('--lr', type=float, default=model_configs.lr, help='Learning rate for optimizer')
    argparser.add_argument('--epochs', type=int, default=model_configs.epochs, help='Number of training epochs')
    argparser.add_argument('--beta_1', type=float, default=model_configs.beta_1, help='Beta_1 parameter for Adam optimizer')
    argparser.add_argument('--beta_2', type=float, default=model_configs.beta_2, help='Beta_2 parameter for Adam optimizer')

    args = argparser.parse_args()

    model_configs.batch_size = args.batch_size
    model_configs.lr = args.lr
    model_configs.epochs = args.epochs
    model_configs.beta_1 = args.beta_1
    model_configs.beta_2 = args.beta_2

    return


if __name__ == "__main__":
    parse_args()
    run = wandb.init(project=params.WANDB_PROJECT, entity=params.ENTITY, job_type="training")
    train_generator, validation_generator = generate_train()
    model = create_model()
    train_model(model, train_generator, validation_generator)
    run.link_model(path="model.keras", registered_model_name="flower_classifier_besthparams")
    run.finish()
    wandb.finish()
