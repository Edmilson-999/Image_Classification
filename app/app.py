import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Inicializar o app Flask
app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.config['UPLOAD_FOLDER'] = 'uploads'

# Carregar o modelo treinado
model = load_model('model/final_model.h5')
img_size = (224, 224)

# Rotas
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)

            # Processar a imagem
            image = load_img(path, target_size=img_size)
            image = img_to_array(image)
            image = preprocess_input(image)
            image = np.expand_dims(image, axis=0)

            # Fazer a predição
            prediction = model.predict(image)[0][0]
            resultado = "Aluno" if prediction > 0.5 else "Outra pessoa"

            return render_template('index.html', prediction=resultado)

    return render_template('index.html')

# Rodar a aplicação
if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
