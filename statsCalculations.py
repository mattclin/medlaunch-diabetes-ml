
from sqlite3 import Timestamp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
from datetime import datetime
import statistics

FILENAME = "/Users/mlwchang/Desktop/MedLaunchProject/Calculating_Stats/Arakali_Sujai_2021-10-21_3Days.csv"

def read_data():
        '''Reads csv data from self.filename as a pandas dataframe.'''

        keep_cols = [ 'Timestamp (YYYY-MM-DDThh:mm:ss)',
                      'Event Subtype',
                      'Glucose Value (mg/dL)',
                      'Insulin Value (u)',
                      'Carb Value (grams)',
                      'Duration (hh:mm:ss)',
                      'Glucose Rate of Change (mg/dL/min)' ]
        '''
        # Add back in if necessary
        ignore_cols = [ 'Device Info', 
                        'Source Device ID', 
                        'Transmitter Time (Long Integer)',
                        'Transmitter ID' ]
        '''
        
        df = pd.read_csv(FILENAME, usecols=keep_cols, parse_dates= ['Timestamp (YYYY-MM-DDThh:mm:ss)'])
        
        df.rename(
            columns=({ 'Timestamp (YYYY-MM-DDThh:mm:ss)': 'Timestamp',
                       'Event Type': 'EventType',
                       'Event Subtype': 'EventSubtype',
                       'Glucose Value (mg/dL)': 'Glucose',
                       'Insulin Value (u)': 'Insulin',
                       'Carb Value (grams)': 'Carb',
                       'Duration (hh:mm:ss)': 'Duration',
                       'Glucose Rate of Change (mg/dL/min)': 'Rate' }), inplace=True)
        return df
# calculate the percentage of time that blood glucose is between 70 and 170 between timestamps passed in
# Also calculates maximum glucose between two timestamps passed in 
# returns a dictionary with various stats
def calculateStats(time1, time2, glucoseDict):
    x = 0
    glucoseList = []
    for item in glucoseDict:
        if (max(item,time1) == item and max(item,time2) == time2):
            glucoseList.append(glucoseDict[item])
            x = x+1
    maxGlucose = glucoseList[0]
    inRange = 0
    
    for i in glucoseList:
        if i > maxGlucose:
            maxGlucose = i
        if i > 70 and i < 170:
            inRange += 1
    inRangePercent = inRange/len(glucoseList) * 100
    avgGlucose = sum(glucoseList)/len(glucoseList)
    stDevGlucose = statistics.stdev(glucoseList)
    statsDict = {"Max": maxGlucose, "inRangePercent" : inRangePercent, "Average": avgGlucose, "Standard_Deviation": stDevGlucose }
    return statsDict
# Returns a percentage between 0 and 100 based on weights passed in and the stats dictionary
def calculateOverallPercentage(statsDict, weights):
    percentage = 0
    if statsDict["Max"] < 170:
        percentage += weights[0] * 100
    elif statsDict["Max"] < 250:
        percentage += weights[0] * 70
    elif statsDict["Max"] < 350:
        percentage += weights[0] * 40
    
    percentage += (statsDict["inRangePercent"] + 10) * weights[1]
    return percentage

# returns a letter grade based on the percentage
def calculateLetterGrade(percentage):
    print(percentage)
    if percentage < 60:
        return "E"
    elif percentage < 70:
        grade = "D"
    elif percentage < 80:
        grade = 'C'
    elif percentage < 90:
        grade = 'B'
    else:
        grade = 'A'
    if (grade == 'A'):
        if percentage % 10 < 3:
            grade += '-'
        if percentage >= 100:
            grade = 'A+'
    elif percentage % 10 < 3:
        grade += '-'
    elif percentage % 10 > 7:
        grade += '+'
    return grade
def main():
    # read data from csv
    df = read_data()

    # make a dictionary using the glucose values
    glucose = pd.Series(df.Glucose.values, index=df.Timestamp).dropna().to_dict()

    # get the keys of the dictionary (these are the time stamps)
    glucose_keys = list(glucose.keys())

    # Calculate the various statistics between the first and last timestamps, returns a dictionary with keys of [maxGlucose, inRangePercent, Average, Standard_Deviation]    
    statsDict = calculateStats(glucose_keys[0], glucose_keys[-1],glucose)

if __name__ == "__main__":
    main()