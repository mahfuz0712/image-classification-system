from src.data_preprocessing import load_data, augment_images
from src.model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore

def train_model():
    # Load and augment data
    x_train, y_train, _, _ = load_data()
    augmented_data = augment_images(x_train, y_train)
    
    # Get the number of classes
    num_classes = y_train.shape[1]
    
    # Build the model
    model = build_model(num_classes)
    
    # Define callbacks
    checkpoint = ModelCheckpoint('../models/best.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    
    # Train the model
    model.fit(
        augmented_data,
        steps_per_epoch=len(augmented_data),
        epochs=25,
        validation_data=(x_train, y_train),
        callbacks=[checkpoint, early_stopping]
    )
    
    print("Training complete. Best model saved as 'best.keras' in models directory.")
    ## best.h5 is deppreacated
