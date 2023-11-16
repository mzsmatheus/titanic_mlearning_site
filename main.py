#imports de python web
from flask import Flask
from flask import render_template, request

#import do arquivo de lógica de machine learning
import machinelearn as ml

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def homepage():
    if request.method == 'POST':
        max_depth = int(request.form['parametros'])
        results = ml.treinamento_dt(max_depth)

        return render_template("home.html", results=results)

    return render_template("home.html", results={})

app.run(debug=True)