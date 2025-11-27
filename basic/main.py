
import pickle
from flask import Flask,request,app, jsonify, render_template
import numpy as np
import pandas as pd
app=Flask(__name__)

## Load the model
regmodel=pickle.load(open('pkl_file','rb'))                     # opening the pickle file

@app.route('/')                                    # route -- will take you somewhere
def home(): 
    return render_template("home.html")             # render the home.html file so we can see it 

@app.route('/predict_api', methods=['post'])            # if anyone sends a request to the URL ending with "predict" run the function
def predict_api():                          
    data=request.json['data']                   # looks at the incoming request and grabs the JSON body
    print(data)
    input=np.array(list(data.values())).reshape(1,-1)           # 
    output=regmodel.predict(input)                      # a regression model that is loaded earlier 
    print(output[0])    
    return jsonify(output[0])           # Access the first item [0],
@app.route('/predict', methods=['POST'])

def predict():                                      # Define predict function
    input=[float(x) for x in request.form.values()] # This line looks at the data
    print(input)
    input_array = np.array(input).reshape(1, -1)       # Converts the list inot Numpy array
    output=regmodel.predict(input_array)[0]            # Calculates the predicitions 
    return render_template("home.html", prediction_test="The predicted mpg value  is {}".format(output))


if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0")



# def main():
#     print("Hello from basic!")


# if __name__ == "__main__":
#     main()
