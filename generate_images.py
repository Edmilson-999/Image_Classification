import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from PIL import Image

# Caminhos
original_dataset_dir = 'dataset'
augmented_dataset_dir = 'augmented_data'
categorias = ['aluno', 'outros']

# Parâmetros
num_aug_por_imagem = 10  # Quantas variações por imagem
img_size = (224, 224)    # Tamanho padrão para redimensionar as imagens

# Gerador de imagens com data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Garantir que a pasta de destino existe
for categoria in categorias:
    os.makedirs(os.path.join(augmented_dataset_dir, categoria), exist_ok=True)

# Para cada categoria (aluno, outros)
for categoria in categorias:
    path_origem = os.path.join(original_dataset_dir, categoria)
    path_destino = os.path.join(augmented_dataset_dir, categoria)
    
    for filename in os.listdir(path_origem):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(path_origem, filename)
            img = load_img(img_path, target_size=img_size)  # Carrega e redimensiona
            x = img_to_array(img)  # Converte em array NumPy
            x = x.reshape((1,) + x.shape)  # Prepara para o gerador

            prefix = os.path.splitext(filename)[0]

            i = 0
            for batch in datagen.flow(x, batch_size=1):
                new_filename = f"{prefix}_aug{i}.jpg"
                array_to_img(batch[0]).save(os.path.join(path_destino, new_filename))
                i += 1
                if i >= num_aug_por_imagem:
                    break
