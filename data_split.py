import os
import shutil
import random

# Caminho das imagens aumentadas
origem_base = 'augmented_data'
destino_base = 'data'

categorias = ['aluno', 'outros']

# Proporções
proporcao_treino = 0.7
proporcao_val = 0.15
proporcao_teste = 0.15

# Garantir pastas de destino
for categoria in categorias:
    for conjunto in ['train', 'val', 'test']:
        path = os.path.join(destino_base, conjunto, categoria)
        os.makedirs(path, exist_ok=True)

# Para cada categoria
for categoria in categorias:
    caminho_origem = os.path.join(origem_base, categoria)
    imagens = [f for f in os.listdir(caminho_origem) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(imagens)

    total = len(imagens)
    n_train = int(total * proporcao_treino)
    n_val = int(total * proporcao_val)

    for i, nome_arquivo in enumerate(imagens):
        caminho_fonte = os.path.join(caminho_origem, nome_arquivo)

        if i < n_train:
            destino = os.path.join(destino_base, 'train', categoria)
        elif i < n_train + n_val:
            destino = os.path.join(destino_base, 'val', categoria)
        else:
            destino = os.path.join(destino_base, 'test', categoria)

        shutil.copy(caminho_fonte, os.path.join(destino, nome_arquivo))
