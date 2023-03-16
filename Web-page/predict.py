import numpy as np
import pickle

# Load the trained model
with open('random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to predict diabetes outcome
def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
    prediction = model.predict(input_data)
    return prediction[0]
