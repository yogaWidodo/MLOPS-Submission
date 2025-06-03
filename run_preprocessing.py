import pandas as pd
import importlib.util
import sys

# Load the module from the filename with spaces
module_name = "automate_script"
file_path = "preprocessing/automate_Yoga.py"
spec = importlib.util.spec_from_file_location(module_name, file_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load module {module_name} from {file_path}")
automate_module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = automate_module
spec.loader.exec_module(automate_module)

def main():
    # Load the raw dataset
    raw_data_path = "preprocessing/Gold Price (2013-2023).csv"
    df = pd.read_csv(raw_data_path)

    # Apply preprocessing
    processed_df = automate_module.transform_gold_data(df)

    # Save the processed dataset
    processed_data_path = "preprocessing/processed_gold_price.csv"
    processed_df.to_csv(processed_data_path, index=False)
    print(f"Processed dataset saved to {processed_data_path}")

if __name__ == "__main__":
    main()
