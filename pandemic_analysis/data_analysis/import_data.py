import pandas as pd 
from data_analysis.models import PandemicData
from datetime import datetime

def import_exel_to_db(file_path):
    sheet_names = pd.ExcelFile(file_path).sheet_names
    print(sheet_names)

    data = pd.read_excel(file_path, sheet_name="all")
    data = data[['Date', 'TotalCases', 'NewCases', 'ICU', 'DEATHS', 'NewDeaths', 'TestTotal', 'PCRperDay']]


    print(data.columns)

    for _, row in data.iterrows():

        if pd.isnull(row['Date']):
            print(f"Skipping row with missing date: {row}")
            continue

        
        date = row['Date']
        
       
        record = PandemicData(
            date = date,
            totalCases=row['TotalCases'],
            newCases=row['NewCases'],
            intenciveCareUnit=row['ICU'],
            deaths=row['DEATHS'],
            newDeaths=row['NewDeaths'],
            totalTests=row['TestTotal'],
            PCRperDay = row['PCRperDay']
        )

        print(record)
        try:
            record.save()
        except Exception as e:
            print(f"Error saving record: {e}")


        
    print("Data import complete")