from flask import Flask,render_template,request,jsonify
from src.Pipeline.prediction_pipeline import PredictPipeline,CustomData

application = Flask(__name__)
app = application


@app.route("/")
def home():
   return  render_template('index.html')


@app.route("/predict",methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
       return  render_template('form.html')
    else:
        data = CustomData(
            age=int(request.form.get('age')),
            experience= int(request.form.get('experience')),
            income=int(request.form.get('income')),
            family=int(request.form.get('family')),
            ccavg=float(request.form.get('ccavg')),
            education=int(request.form.get('education')),
            mortgage=float(request.form.get('mortgage')),
            securitiesAcc=int(request.form.get('securitiesacc')),
            CdAcc=int(request.form.get('cdacc')),
            Online=int(request.form.get('online')),
            Creditcard=int(request.form.get('creditcard'))
        )

        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        predict = predict_pipeline.predict(final_new_data)
        print(f"predicted value is : {int(predict[0])}")
        if int(predict[0]==0):
            result = "Customer is not Eligible to receive a Bank Loan"
        else:
            result = "Customer is Eligible to receive a Bank Loan"
        return jsonify({'result':result})





if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=8000)