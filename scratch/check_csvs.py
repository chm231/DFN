import pandas as pd
df1 = pd.read_csv("storage/output/ground_truth_traces.csv")
print("ground_truth_traces.csv columns:", df1.columns.tolist())
if len(df1) > 0:
    print("row 0:", df1.iloc[0].to_dict())

df2 = pd.read_csv("storage/output/ground_truth_traces_with_normals.csv")
print("ground_truth_traces_with_normals.csv columns:", df2.columns.tolist())
if len(df2) > 0:
    print("row 0:", df2.iloc[0].to_dict())
