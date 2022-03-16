from flask import Flask, render_template, request
from flask import request
from statsCalculations import calculateLetterGrade, read_data, fullProcessing 
from datetime import datetime
from sqlite3 import Timestamp
import os
DATA_NAME = "Data"
VAR_DICT = {}

app = Flask(__name__)


@app.route('/')
def index():
    return 'Index Page'

@app.route("/hello")
def hello_world():
    return "<p>Hello, World!</p>" + calculateLetterGrade(85)


# @app.route("/read_data")
# def processFile():
#     data = read_data(DATA_NAME)
#     return (fullProcessing(DATA_NAME,0,-1))

@app.route("/form/")
def form():
    return render_template('form.html')
 
@app.route('/upload', methods = ['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['File']
        # save the file
        f.save(f.filename)

        # save the dates submitted by the user
        date1 = request.form.get("date1") + ":" + request.form.get('time1')
        date1 = datetime.strptime(date1, '%Y-%m-%d:%H:%M')
        date2 = request.form.get("date2") + ":" + request.form.get('time2')
        date2 = datetime.strptime(date2, '%Y-%m-%d:%H:%M')

        # add variables to global dictionary if needed in other routes
        VAR_DICT.update({"Filename": f.filename})
        VAR_DICT.update({"date1": date1})
        VAR_DICT.update({"date2": date2})
        print(request.form.get("Date"))
        print(request.form.get('Blank'))
        print(VAR_DICT['date1'])
        print(VAR_DICT['date2'])

        # returns stats dict (keys: Max, inRangePercent, avgGlucose, stDevGlucose)
        stats,grade = fullProcessing(f.filename,date1,date2)
        print(stats["inRangePercent"])
        return grade + "Data and file saved successfully"


if __name__==('__main__'):
    app.run(debug=True)