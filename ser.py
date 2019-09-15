from flask import Flask
import routf
ml =Flask("ML")
ml.register_blueprint(routf.grec_predictor)
ml.run(host ="0.0.0.0" ,port=5005,debug=True)