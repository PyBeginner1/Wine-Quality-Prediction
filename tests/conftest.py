import pytest
import yaml
import os
import json 
from pathlib import Path

@pytest.fixture
def config(config_path="params.yaml"):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(config_path)
    return config


@pytest.fixture
def config(schema_path=Path("prediction_service\schema_in.json")):
    with open(schema_path) as json_file:
        config = json.load(schema_path)
    return config