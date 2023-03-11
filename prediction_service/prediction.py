import yaml
import joblib
import numpy as np
import os 
import json
import shutil
from pathlib import Path
import shutil

params_path = "params.yaml"
schema_path = os.path.join('prediction_service', 'schema_in.json')

source_file = 'notebooks\schema_in.json'
destination_folder = schema_path
if not os.path.exists(destination_folder):
    shutil.copy(source_file, destination_folder)

current_dir = os.getcwd()
src_path = os.path.join(current_dir, Path('saved_models/model.joblib'))
dest_path = os.path.join(current_dir, Path('prediction_service/model/model.joblib'))
check_dir = os.path.join(current_dir, Path('prediction_service/model'))

class NotInRange(Exception):
    def __init__(self, message='Values entered not in range'):
        self.message = message
        super().__init__(self.message)

#For api    
class NotInColumn(Exception):
    def __init__(self, message='Not in columns'):
        self.message = message
        super().__init__(self.message)


def form_response(dict_request):
    if validate_input(dict_request):
        data = dict_request.values()
        data = [list(map(float, data))]
        response = prediction(data)
        return response


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def prediction(data):
    try:
        config = read_params(params_path)
        if not os.listdir(check_dir):
            shutil.copy(src_path, dest_path)
        model_dir = config['webapp_model_dir']
        model = joblib.load(model_dir)
        prediction = model.predict(data)

        try:    
            if 3 <= prediction[0] <= 8:
                return int(prediction[0])
            else:
                raise NotInRange
        except Exception as e:
            return "Unexpected result"
    except Exception as e:
        raise e
    

def api_response(request):
    try:
        data = np.array(list(request.form.values()))
        response = prediction(data)
        response = {'response': response}
        return response
    except Exception as e:
        raise e
    

def get_schema(schema_path=schema_path):
    with open(schema_path) as json_file:
        schema = json.load(json_file)
    return schema


def validate_input(dict_request):
    def _validate_cols(col):
        schema = get_schema()
        actual_cols = schema.keys()
        if col not in actual_cols:
            raise NotInColumn
        
    def _validate_values(col, val):
        schema = get_schema()

        if not (schema[col]["min"] <= float(dict_request[col]) <= schema[col]["max"]):
            raise NotInRange
        
    for col, val in dict_request.items():
        _validate_cols(col)
        _validate_values(col,val)

    return True
    