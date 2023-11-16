#imports de python web
from flask import Flask
from flask import render_template

#import do arquivo de lógica de machine learning
import machinelearn as ml


app = Flask(__name__)

@app.route("/")
def homepage():
    # Chama a função que executa o treinamento e teste do modelo
    results = ml.treinamentoteste()

    return render_template("home.html", results=results)

app.run(debug=True)