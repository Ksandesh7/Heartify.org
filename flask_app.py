from flask import Flask, render_template, request, send_file, send_from_directory
import pickle
import numpy as np

app = Flask(__name__, static_url_path='', static_folder='static',template_folder='templates')

#LR
model1 = pickle.load(open('model1.pkl', 'rb'))
#Dtree
model2 = pickle.load(open('model2.pkl', 'rb'))
#RF
model3 = pickle.load(open('model3.pkl', 'rb'))
#KNN
model4 = pickle.load(open('model4.pkl', 'rb'))




# Define the home route
@app.route('/')
def home():
    return send_file('index.html')



@app.route('/templates/predict_form.html', methods=['GET', 'POST'])
def prediction():
    return render_template('predict_form.html')

@app.route('/templates/suggest.html', methods=['GET', 'POST'])
def suggest():
    return render_template('suggest.html')

@app.route('/templates/result.html', methods=['GET', 'POST'])
def result():
    return render_template('result.html')



# Define the prediction route
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]

    #D Tree
    prediction2 = model2.predict_proba(final)
    output2 = '{0:.{1}f}'.format(prediction2[0][1], 2)

    #RF
    prediction3 = model3.predict_proba(final)
    output3 = '{0:.{1}f}'.format(prediction3[0][1], 2)

    #kNN
    prediction4 = model4.predict_proba(final)
    output4 = '{0:.{1}f}'.format(prediction4[0][1], 2)

    #LR
    prediction1 = model1.predict_proba(final)
    output1 = '{0:.{1}f}'.format(prediction1[0][1], 2)

    return render_template('result.html',pred1=float(output1), pred2=float(output2), pred3=float(output3), pred4=float(output4))



if __name__ == '__main__':
    app.run(debug=True)






















































































































































    # if request.method == 'POST':

    #     age = int(request.form['age'])
    #     sex = request.form.get('sex')
    #     cp = request.form.get('cp')
    #     trestbps = int(request.form['trestbps'])
    #     chol = int(request.form['chol'])
    #     fbs = request.form.get('fbs')
    #     restecg = int(request.form['restecg'])
    #     thalach = int(request.form['thalach'])
    #     exang = request.form.get('exang')
    #     oldpeak = float(request.form['oldpeak'])
    #     slope = request.form.get('slope')
    #     ca = int(request.form['ca'])
    #     thal = request.form.get('thal')
        
    #     data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    #     my_prediction = model.predict(data)
        
    #     output = round(my_prediction[0], 2)

    #     if output > 0.5:
    #         return render_template('predict_form.html', pred='Your heart is in danger.\n Probability of occuring heart disease is {}'.format(output))
    #     else:
    #         return render_template('predict_form.html', pred='Your heart is safe.\n Probability of occuring heart disease is {}'.format(output))