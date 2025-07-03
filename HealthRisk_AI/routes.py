from flask import render_template,url_for
from HealthRisk_AI import app
from HealthRisk_AI.forms import InputForm

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
    form = InputForm()
    #if form.validate_on_submit():

    return render_template('predict.html', form=form)