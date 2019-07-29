from flask import Flask
import routes1
ml =Flask("ML")
ml.register_blueprint(routes1.clf_predictor)
ml.run(host ="0.0.0.0" ,port=5005,debug=True)