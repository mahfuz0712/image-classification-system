from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.models import Model, Sequential # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

def build_model(num_classes):
    # Load ResNet50 without the top layer (include_top=False)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Global Average Pooling
    x = Dropout(0.5)(x)  # Add Dropout for regularization
    x = Dense(128, activation='relu')(x)  # Fully connected layer
    predictions = Dense(num_classes, activation='softmax')(x)  # Final output layer
    
    # Combine base model and custom layers
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze base model layers for transfer learning
    for layer in base_model.layers:
        layer.trainable = False
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
