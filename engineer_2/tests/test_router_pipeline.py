import json
import os

def test_config_generation():
    config_path = "engineer_2/configs/sample_labeled_queries.json"
    assert os.path.exists(config_path), "Configuration file was not generated"
    
    with open(config_path, "r") as f:
        data = json.load(f)
        
    assert len(data) > 0, "Generated data is empty"
    for item in data:
        assert "query" in item, "Missing 'query' key in labeled output"
        assert "label" in item, "Missing 'label' key in labeled output"
        assert item["label"] in [0, 1, 2], f"Invalid label value: {item['label']}"
        
    print("✓ All Engineer 2 integrity unit tests passed!")

if __name__ == "__main__":
    test_config_generation()
