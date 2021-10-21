import pickle

from flask import Flask, render_template, request
from flask_cors import cross_origin
import numpy as np

application = Flask(__name__) # initializing a flask app

@application.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")


@application.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #Index(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
            #      'PTRATIO', 'B', 'LSTAT', 'Price'],
            #  reading the inputs given by the user

            crim=float(request.form['CRIM'])
            zn = float(request.form['ZN'])
            indus = float(request.form['INDUS'])
            nox = float(request.form['NOX'])
            rm = float(request.form['RM'])
            age = float(request.form['AGE'])
            dis = float(request.form['DIS'])
            rad = float(request.form['RAD'])
            tax = float(request.form['TAX'])
            ptratio = float(request.form['PTRATIO'])
            b = float(request.form['B'])
            lstat = float(request.form['LSTAT'])
            chas = int(request.form['CHAS'])
            model_fileName = 'regression_model.pickle'
            scaler_fileName='scaler_transformation.pickle'
            loaded_model = pickle.load(open(model_fileName, 'rb')) # loading the model file from the storage
            loaded_scalar = pickle.load(open(scaler_fileName, 'rb')) #
            # predictions using the loaded model file
            temp_scaled=loaded_scalar.transform([[crim,zn,indus,nox,rm,age,dis,rad,tax,ptratio,b,lstat]])
            print("Initial scaling",temp_scaled)
            temp_scaled = np.delete(temp_scaled, [3, 7, 8], 1)
            # Added the CHAS value as 0 as shown below:
            temp_scaled = np.column_stack((temp_scaled, chas))
            print("Final one",temp_scaled)
            prediction=loaded_model.predict(temp_scaled)
            print('prediction is', prediction)
            return render_template('results.html', prediction=round(prediction[0], 2))
        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    application.run(debug=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
