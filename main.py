#imports de python web
from flask import Flask
from flask import render_template, request

#import do arquivo de lógica de machine learning
import machinelearn as ml

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def homepage():
    if request.method == 'POST':
        
        if 'classificador' in request.form:
            classificador = request.form['classificador'] 
            param1 = int(request.form.get('parametro1', 1))
            param2 = int(request.form.get('parametro2', 2)) if request.form.get('parametro2') else 2
            param3 = int(request.form.get('parametro3', 2)) if request.form.get('parametro3') else 2

            if classificador == 'KNN':
                results = ml.treinamentoteste('KNN', param1, param2, param3)
            elif classificador == 'SVC':
                results = ml.treinamentoteste('SVC', param1, param2, param3)
            elif classificador == 'DT':
                results = ml.treinamentoteste('DT', param1, param2, param3)
            elif classificador == 'RF':
                results = ml.treinamentoteste('RF', param1, param2, param3)
            elif classificador == 'GBM':
                results = ml.treinamentoteste('GBM', param1, param2, param3)
            else:
                results = {}
            
            return render_template("home.html", size_str=results['tamanho:'], results=results)

        else:

            return "Classificador não selecionado."

    return render_template("home.html", results={})


app.run(debug=True)