from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your trained models
resnet_model = load_model('resnet_model.h5')
vggnet_model = load_model('vggnet_model.h5')
cnn_model = load_model('cnn_model.h5')

# Define route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Define route for model prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        

        # Make predictions using different models
        resnet_pred = resnet_model.predict(img_array)
        vggnet_pred = vggnet_model.predict(img_array)
        cnn_pred = cnn_model.predict(img_array)

        # Assuming you want to return class labels along with predictions
        resnet_label = np.argmax(resnet_pred)
        vggnet_label = np.argmax(vggnet_pred)
        cnn_label = np.argmax(cnn_pred)

        return jsonify({
            'resnet_prediction': resnet_label,
            'vggnet_prediction': vggnet_label,
            'cnn_prediction': cnn_label
        })



if __name__ == '__main__':
    app.run(debug=True)
