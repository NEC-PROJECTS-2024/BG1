import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
app = Flask(__name__)
model = pickle.load(open('model_svc.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/home')
def default():
    return render_template('home.html')
@app.route('/risk', methods=['POST', 'GET'])
def risk():
    answers = {'q1': '', 'q2': '', 'q3': '', 'q4': '', 'q5': '', 'q6': '', 'q7': '', 'q8': ''}

    if request.method == 'POST':
        # Getting the answers from the form
        answers['q1'] = request.form.get('q1')
        answers['q2'] = request.form.get('q2')
        answers['q3'] = request.form.get('q3')
        answers['q4'] = request.form.get('q4')
        answers['q5'] = request.form.get('q5')
        answers['q6'] = request.form.get('q6')
        answers['q7'] = request.form.get('q7')
        answers['q8'] = request.form.get('q8')

        # Counting the number of 'yes' answers
        yes_count = sum(1 for answer in answers.values() if answer == 'yes')

        # Determining the danger level
        if yes_count == 0:
            danger_level = 'No Danger'
        elif yes_count < 5:
            danger_level = 'Mild Danger'
        else:
            danger_level = 'Heavy Danger'

        return render_template('risk.html', dangerLevelMessage=danger_level, answers=answers)

    return render_template('risk.html', dangerLevelMessage=None, answers=answers)

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method=='POST':
        input_features = [float(x) for x in request.form.values()]
        features_value = np.array([input_features])
        features_value=features_value.reshape(1,-1)
        if not hasattr(ss,'mean_'):
            ss.fit(features_value)
        else:
            ss.partial_fit(features_value)
        scaled_input=ss.transform(features_value)
        
        features_name = ['radius_mean', 'texture_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean',
            'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se',  'smoothness_se',
        'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
        'fractal_dimension_se',  
        'smoothness_worst',
        'compactness_worst', 'concavity_worst', 
        'symmetry_worst', 'fractal_dimension_worst']
        
        
        output = model.predict(scaled_input)
            
        if output == 1:
            res_val = "** breast cancer **"
        else:
            res_val = "no breast cancer"
            

        return render_template('index.html', prediction_text='Patient has {}'.format(res_val))
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)