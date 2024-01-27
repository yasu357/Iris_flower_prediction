from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

model = keras.models.load_model('iris_model.h5')
le = LabelEncoder()

sc = StandardScaler()

training_data_path = r'C:\Users\shaik\Downloads\Irisdataset.csv' 
training_data = pd.read_csv(training_data_path)


sc.fit(training_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])
le.fit(training_data['Species'])
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
  
        features = [float(x) for x in request.form.values()]
        input_data = pd.DataFrame([features], columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

        input_data = sc.transform(input_data)

        predictions = model.predict(input_data)
        predicted_class = le.inverse_transform(np.argmax(predictions, axis=-1))[0]

        return render_template('index.html', prediction_text=f'The predicted species is: {predicted_class}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
