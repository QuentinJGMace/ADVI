import numpy as np
import ast
from pdb import set_trace
from tqdm import tqdm
import pandas as pd

FILE_NAME = 'data/test.csv'

data_df = pd.read_csv('FILE_NAME')
print(data_df.columns)

target_shape = 50

x_data = []

print("Interpolating the arrays...")

# Function to interpolate and resize the arrays
for i, array in tqdm(enumerate(data_df["POLYLINE"])):
    # Convert the string representation of an array into an actual array
    array = np.array(ast.literal_eval(array))
    
    # Extract x and y coordinates from the array of tuples
    if data_df["MISSING_DATA"][i] == "True":
        continue
    try:
        x = array[:, 0]
        y = array[:, 1]
    except:
        continue
    
    # Interpolate the x and y coordinates to the target shape
    interpolated_x = np.interp(np.linspace(0, 1, target_shape), np.linspace(0, 1, len(x)), x)
    interpolated_y = np.interp(np.linspace(0, 1, target_shape), np.linspace(0, 1, len(y)), y)
    
    # Combine the interpolated x and y coordinates into an array of tuples
    interpolated_array = np.column_stack((interpolated_x, interpolated_y))
    
    # Append the interpolated array to the list
    x_data.append(interpolated_array)

print("Turning into array...")

# Convert the list x_train into an array
x_data_array = np.array(x_data)

print("Saving into npz file...")

# Save the array into a .npz file
np.savez('data/x_test_array.npz', x_train_array=x_data_array)