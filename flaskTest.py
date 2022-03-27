from flask import Flask, render_template, request
from statsCalculations import calculateLetterGrade, read_data, fullProcessing, graphData 
from datetime import datetime



DATA_NAME = "Data"
VAR_DICT = {}

app = Flask(__name__)


# Home Page
@app.route('/')

def index1():
    return render_template('/medlaunch-diabetes-website/index.html')

@app.route("/calculator")
def calculator1():
    return render_template('/medlaunch-diabetes-website/calculator.html')

@app.route("/quickfacts")
def quickfacts1():
    return render_template('/medlaunch-diabetes-website/quickfacts.html')

@app.route("/treatments")
def treatments1():
    return render_template('/medlaunch-diabetes-website/treatments.html')

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
        # VAR_DICT.update({"Filename": f.filename})
        # VAR_DICT.update({"date1": date1})
        # VAR_DICT.update({"date2": date2})
        # print(request.form.get("Date"))
        # print(request.form.get('Blank'))
        # print(VAR_DICT['date1'])
        # print(VAR_DICT['date2'])

        # returns stats dict (keys: Max, inRangePercent, Average, Standard_Deviation)
        
        #graphName = graphData(f.filename)
        stats,grade = fullProcessing(f.filename,date1,date2)
        inRange = stats['inRangePercent']
        
        return render_template('data.html', gradeValue = grade, inRangePercent = round(inRange,2), mean = round(stats['Average'],2), Max = stats['Max'], Deviation = round(stats['Standard_Deviation'],2))



if __name__==('__main__'):
    app.run(debug=True)