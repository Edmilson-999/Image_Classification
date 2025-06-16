
# Classificação de Imagens de Aluno com Deep Learning e Flask

Este projeto consiste num sistema inteligente capaz de **classificar uma imagem como sendo de um aluno específico ou de outra pessoa**, utilizando técnicas de Visão Computacional, Transfer Learning e uma interface web simples feita com Flask.

---

##  Funcionalidades

- Classificação binária de imagens: **"Aluno"** ou **"Outra pessoa"**.
- Treinamento com rede neural convolucional baseada em **Transfer Learning (MobileNet)**.
- Aplicação de técnicas de **regularização**, como **Dropout** e **EarlyStopping**.
- Sistema de **upload de imagem** com frontend em HTML e CSS.
- Interface com Flask para predição e exibição do resultado.
- **Data Augmentation** para expandir o dataset.

---

##  Estrutura do Projeto

```
 Image_Classification
├── app/
│   └── app.py                # Backend Flask
├── templates/
│   └── index.html            # Frontend (HTML)
├── static/
│   ├── style.css             # Estilo da interface
├── uploads/                  # Onde imagens carregadas são armazenadas (temporariamente)
├── dataset/                  # Imagens originais
│   ├── aluno/
│   └── outros/
├── augmented_data/          # Imagens aumentadas
│   ├── aluno/
│   └── outros/
├── data/                    # Dados divididos
│   ├── train/
│   ├── val/
│   └── test/
├── model/
│   ├── final_model.h5        # Modelo treinado
│   └── mobilenet_model.h5    # Modelo base transferido
├── generate_images.py        # Script de Data Augmentation
├── data_split.py             # Script de divisão de dados
├── train_model.py            # Script de treino
└── README.md                 # Este ficheiro
```

---

##  Como Executar o Projeto

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

> **Bibliotecas utilizadas**: `flask`, `tensorflow`, `keras`, `numpy`, `Pillow`

### 2. Gerar imagens aumentadas

```bash
python generate_images.py
```

### 3. Dividir o dataset

```bash
python data_split.py
```

### 4. Treinar o modelo

```bash
python train_model.py
```

### 5. Iniciar o servidor Flask

```bash
cd app
python app.py
```

Acesse em: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧠 Tecnologias e Técnicas

- **Linguagem**: Python
- **Framework Web**: Flask
- **Deep Learning**: TensorFlow / Keras
- **Transfer Learning**: MobileNet
- **Pré-processamento**: Redimensionamento, normalização
- **Regularização**: Dropout, EarlyStopping
- **Augmentation**: Horizontal flip, zoom, brightness

---

##  Possíveis Melhorias Futuras

- Armazenamento de históricos de predição.
- Treinamento incremental (adicionar mais alunos).
- Usar camadas adicionais para aumentar precisão.
- Exportar resultados para CSV.

---

## 👨‍💻 Autor

**Edmilson Alves**  
Estudante de Engenharia Informática  
Universidade do Mindelo, Cabo Verde  

---

## 📄 Licença

Este projeto é de uso educacional e livre para fins acadêmicos.
