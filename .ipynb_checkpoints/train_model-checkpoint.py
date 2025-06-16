import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models, callbacks

# Parâmetros
img_size = (224, 224)
batch_size = 32
epochs = 20

# Caminhos dos dados
data_dir = 'data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

# Geradores de dados com pré-processamento
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_gen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
val_data = val_gen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
test_data = test_gen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False)

# Carregar a base pré-treinada do MobileNetV2
base_model = MobileNetV2(input_shape=img_size + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  # Congelar as camadas convolucionais

# Construir o modelo com camadas personalizadas
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # Saída binária
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks: salvar modelo e parar cedo se não melhorar
checkpoint_cb = callbacks.ModelCheckpoint('model/mobilenet_model.h5', save_best_only=True)
earlystop_cb = callbacks.EarlyStopping(patience=5, restore_best_weights=True)

# Treinar o modelo
history = model.fit(
    train_data,
    epochs=epochs,
    validation_data=val_data,
    callbacks=[checkpoint_cb, earlystop_cb]
)

# Avaliar no conjunto de teste
loss, acc = model.evaluate(test_data)
print(f"\nTest Accuracy: {acc:.2f}")

# Salvar versão final do modelo
model.save('model/final_model.h5')
