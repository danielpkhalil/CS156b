import pandas as pd
import os

# Read the CSV file
df = pd.read_csv('/mnt/c/Users/danie/Downloads/CS156b/train2023.csv')

# Replace missing values with 0.0 (if not already done)
df = df.fillna(0.0)

# Get the list of available PIDs in the 'train' directory
train_dir = '/mnt/c/Users/danie/Downloads/CS156b/train'  # Replace with the actual path to the 'train' directory
available_pids = [d.split('pid')[1] for d in os.listdir(train_dir) if d.startswith('pid')]

# Extract the PID part from the 'Path' column
df['PID'] = df['Path'].str.extract(r'train/pid(\d+)', expand=False)

# Filter the DataFrame to keep only rows with available PIDs
df = df[df['PID'].isin(available_pids)]

# Drop the 'PID' column if not needed
df = df.drop('PID', axis=1)

# Save the modified data to a new CSV file
df.to_csv('newtrain2023.csv', index=False)