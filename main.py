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
            max_depth = int(request.form.get('parametros', 0))

            if classificador == 'KNN':
                results = ml.treinamentoteste('KNN', max_depth)
            elif classificador == 'SVC':
                results = ml.treinamentoteste('SVC', max_depth)
            elif classificador == 'DT':
                results = ml.treinamentoteste('DT', max_depth)
            elif classificador == 'RF':
                results = ml.treinamentoteste('RF', max_depth)
            else:
                results = {}
            
            return render_template("home.html", results=results)

        else:

            return "Classificador não selecionado."

    return render_template("home.html", results={})


app.run(debug=True)