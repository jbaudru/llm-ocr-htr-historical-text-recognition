import os
import pandas as pd
import json

def convert_excel_to_json(file_path, output_folder):
    # Read the Excel file with two header rows -- Modify this if your Excel file has a different structure
    excel_data = pd.read_excel(file_path, header=[0, 1])

    # Convert the DataFrame to a nested dictionary
    nested_dict = {}
    for idx, row in excel_data.iterrows():
        nested_row = {}
        for col in excel_data.columns:
            header1, header2 = col
            if header1 not in nested_row:
                nested_row[header1] = {}
            nested_row[header1][header2] = row[col]
        nested_dict[idx] = nested_row

    # Convert the nested dictionary to JSON
    json_data = json.dumps(list(nested_dict.values()), indent=4, ensure_ascii=False)
    
    # Extract file name without extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    json_file_path = os.path.join(output_folder, f"{file_name}.json")
    
    # Save JSON data to a file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json_file.write(json_data)

    print(f"Conversion complete. JSON data saved to '{json_file_path}'.")

input_folder = os.path.join(os.getcwd(), 'data/transcriptions')
output_folder = os.path.join(os.getcwd(), 'data/json_transcriptions')

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each Excel file in the folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.xlsx'):
        file_path = os.path.join(input_folder, file_name)
        convert_excel_to_json(file_path, output_folder)

print("All files have been processed.")
