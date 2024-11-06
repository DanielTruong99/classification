import pandas as pd
import os

root_path = os.getcwd() + '/raw_data/Nov05'

# Create the processed_data directory if it doesn't exist
processed_dir = os.path.join(root_path, 'processed_data')
os.makedirs(processed_dir, exist_ok=True)

# Loop through all files in the current folder
for filename in os.listdir(root_path):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(root_path, filename)
        
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Add new columns based on the conditions
        df['is_on_1'] = df['r1'].apply(lambda x: 1 if x == 1 else 0)
        df['is_on_2'] = df['r2'].apply(lambda x: 1 if x == 1 else 0)
        df['is_on_3'] = df['r3'].apply(lambda x: 1 if x == 1 else 0)
        df['is_on_4'] = df['r4'].apply(lambda x: 1 if x == 1 else 0)

        # Remove rows where all is_on_1, is_on_2, is_on_3, and is_on_4 are 0
        df = df[~((df['is_on_1'] == 0) & (df['is_on_2'] == 0) & (df['is_on_3'] == 0) & (df['is_on_4'] == 0))]

        # Check if there is any row with only one column in is_on_1, is_on_2, is_on_3, is_on_4 having value 1
        for index, row in df.iterrows():
            is_on_columns = [row['is_on_1'], row['is_on_2'], row['is_on_3'], row['is_on_4']]
            if is_on_columns.count(1) == 1:
                # Find the column with the largest value among r1, r2, r3, r4 except the one already having is_on_ as 1
                r_values = {'r1': row['r1'], 'r2': row['r2'], 'r3': row['r3'], 'r4': row['r4']}
                for i, is_on in enumerate(is_on_columns):
                    if is_on == 1:
                        del r_values[f'r{i+1}']
                        break
                max_r_column = max(r_values, key=r_values.get)
                df.at[index, f'is_on_{max_r_column[-1]}'] = 1

        
        # Save the updated DataFrame to the processed_data directory
        processed_file_path = os.path.join(processed_dir, filename)
        df.to_excel(processed_file_path, index=False)