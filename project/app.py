from flask import Flask, render_template, request
import joblib
import pandas as pd
app = Flask(__name__)
model = joblib.load('yield_model.pkl')  # Trained model
area_encoder = joblib.load('area_encoder.pkl')  # Encoded 'Area'
item_encoder = joblib.load('item_encoder.pkl')  # Encoded 'Item'
# Get labels for dropdowns
area_labels = list(area_encoder.classes_)
item_labels = list(item_encoder.classes_)
@app.route('/')
def home():
    return render_template('index.html', areas=area_labels, items=item_labels)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        area = request.form['area']
        item = request.form['item']
        year = int(request.form['year'])
        rainfall = float(request.form['rainfall'])
        pesticides = float(request.form['pesticides'])
        temperature = float(request.form['temperature'])
        # Encode Area and Item using the trained encoders
        area_encoded = area_encoder.transform([area])[0]
        item_encoded = item_encoder.transform([item])[0]
        # Prepare the input DataFrame with all 6 features
        input_data = pd.DataFrame([[area_encoded, item_encoded, year, rainfall, pesticides, temperature]],
                                  columns=['Area', 'Item', 'Year', 'average_rain_fall_mm_per_year','pesticides_tonnes', 'avg_temp'])
        # Predict the crop yield using the trained model
        prediction = model.predict(input_data)[0]
        return render_template('index.html', prediction=round(prediction, 2),
                               areas=area_labels, items=item_labels)
    except Exception as e:
        return f"❌ Error: {str(e)}"
if __name__ == '__main__':
    app.run(debug=True)