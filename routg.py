from flask import Flask, request, jsonify
import pickle
import sys
sys.setrecursionlimit(100000)
from ipynb.fs.full.recommendation import grec # importing the function grec 
app = Flask(__name__)
si=pickle.load(open("t","rb"))   #loading sigmoid
@app.route("/predict",methods=['POST'])  
def addRating():
    content = request.get_json()    # to get the data 
    predstring = content["input"]
    status = grec(predstring)       #passing the data to the function
    return jsonify({
        "the ans is ":str(status)      # to dispalya the ans in the post 
    })
# Running the server in localhost:5000 
if __name__ == '__main__':     
    app.run(debug=True, host='0.0.0.0', port=5000)