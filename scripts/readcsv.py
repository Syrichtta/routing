import pandas as pd
import ast

# Load CSV into DataFrame
df = pd.read_csv('shortest_paths.csv')

# Parse the 'path' column into actual lists of coordinates
df['path'] = df['path'].apply(ast.literal_eval)

# Example: print the first few rows to check the data
print(df.head())

# You can also extract individual columns like 'start_node', 'destination', etc.
print(df['start_node'])
print(df['destination'])