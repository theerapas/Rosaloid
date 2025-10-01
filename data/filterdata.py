import pandas as pd

# Load the CSV file
file_path = "GFP_AEQVI_Sarkisyan_2016.csv"
df = pd.read_csv(file_path)

# Filter rows where "mutant" column does not contain ":"
single_mut_df = df[~df["mutant"].astype(str).str.contains(":")]

# Save filtered data to new CSV
output_path = "GFP_single_mutation.csv"
single_mut_df.to_csv(output_path, index=False)
