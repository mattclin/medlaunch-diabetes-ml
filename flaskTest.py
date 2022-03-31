from flask import Flask, render_template, request
from statsCalculations import calculateLetterGrade, read_data, fullProcessing, GraphData
from datetime import datetime
import os
import uuid
import glob
import sys


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

        dir_name = os.path.dirname(app.instance_path)
        saved_files = glob.glob(os.path.join(dir_name, '/static/uploads/'), recursive=True)
        print('SAVED', saved_files, file=sys.stderr)
        for item in saved_files:
            print(item, file=sys.stderr)
            if item.endswith(".csv"):
                os.remove(os.path.join(dir_name, '/static/uploads/', item))
            if item.startswith('plot_') and item.endswith('.png'):
                os.remove(os.path.join(dir_name, '/static/uploads/', item))

        f = request.files['File']
        # save the file
        f.save(os.path.join(dir_name, 'static/uploads/', f.filename))

        f.filename = 'static/uploads/' + f.filename

        # save the dates submitted by the user
        date1 = request.form.get("date1")
        date1 = datetime.strptime(date1, '%Y-%m-%d')
        # date2 = request.form.get("date2") + ":" + request.form.get('time2')
        # date2 = datetime.strptime(date2, '%Y-%m-%d:%H:%M')

        stats,grade = fullProcessing(f.filename,date1)
        inRange = stats['inRangePercent']

        image_name = 'plot_' + str(uuid.uuid4().hex) + '.png'
        new_graph = GraphData(f.filename, image_name, date1)
        plt = new_graph.graph()
        plt.savefig('static/uploads/' + image_name)

        plot_path = 'static/uploads/' + image_name

        # graphName = new_graph.imagename #os.path.join(dir_name, image)
        return render_template('/medlaunch-diabetes-website/results.html', 
                gradeValue = grade, 
                inRangePercent = round(inRange,2), 
                mean = round(stats['Average'],2), 
                Max = stats['Max'], 
                Deviation = round(stats['Standard_Deviation'],2),
                plot = plot_path)

if __name__==('__main__'):
    app.run(debug=True)