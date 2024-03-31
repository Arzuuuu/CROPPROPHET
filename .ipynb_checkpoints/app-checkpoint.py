from flask import Flask, request, render_template
import numpy as np
import pandas
import sklearn
import pickle


model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")
@app.route("/predict", methods=['POST'])
def predict():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosphorus'])
    k = int(request.form['Potassium'])
    temperature = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, k, temperature, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    prediction = model.predict(single_pred)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]  # Fixed typo here
        result = f"{crop} is the best crop to be cultivated"
        print(result)  # Printing the result for debugging
    else:
        result = "Sorry, we are not able to recommend a proper crop for this environment"

    return render_template('index.html', result=result)  # Passing the result to the template
