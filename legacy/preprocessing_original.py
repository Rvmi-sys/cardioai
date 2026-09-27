import csv

#LOAD THE DATASET
filename = "heart_disease_uci.csv"

with open(filename) as f:
    reader = csv.DictReader(f)
    data = list(reader)

print("Number of rows loaded:", len(data))
print("Columns:", list(data[0].keys()))

# DROP COLUMNS WE DON'T NEED

columns_to_drop = ["id", "dataset"]

for row in data:
    for col in columns_to_drop:
        del row[col]

print("\nColumns after dropping:", list(data[0].keys()))

# HANDLE MISSING VALUES
def get_median(values):
    sorted_values = sorted(values)
    n = len(sorted_values)
    middle = n // 2
    if n % 2 == 0:
        return (sorted_values[middle - 1] + sorted_values[middle]) / 2
    else:
        return sorted_values[middle]

# Helper function: get the most common value in a list
def get_mode(values):
    counts = {}
    for v in values:
        if v in counts:
            counts[v] += 1
        else:
            counts[v] = 1
    return max(counts, key=counts.get)

# Fill missing values in number columns using the median
number_columns = ["trestbps", "chol", "thalch", "oldpeak"]

for col in number_columns:
    valid_values = []
    for row in data:
        if row[col] != "" and row[col] is not None:
            valid_values.append(float(row[col]))
    median = get_median(valid_values)

    for row in data:
        if row[col] == "" or row[col] is None:
            row[col] = str(median)

    print(f"  {col}: filled missing values with median = {median}")

# Fill missing values in category columns using the mode (most common value)
category_columns = ["fbs", "restecg", "exang", "slope", "ca", "thal"]

for col in category_columns:
    valid_values = []
    for row in data:
        if row[col] != "" and row[col] is not None:
            valid_values.append(row[col])
    mode = get_mode(valid_values)

    for row in data:
        if row[col] == "" or row[col] is None:
            row[col] = mode

    print(f"  {col}: filled missing values with mode = {mode}")

# FIX IMPOSSIBLE CHOLESTEROL VALUES

valid_chol = []
for row in data:
    if float(row["chol"]) > 0:
        valid_chol.append(float(row["chol"]))

chol_median = get_median(valid_chol)

fixed = 0
for row in data:
    if float(row["chol"]) == 0:
        row["chol"] = str(chol_median)
        fixed += 1

print(f"\nFixed {fixed} zero cholesterol value(s) with median = {chol_median}")

# SIMPLIFY THE TARGET COLUMN

for row in data:
    if int(float(row["num"])) > 0:
        row["target"] = "1"
    else:
        row["target"] = "0"
    del row["num"]

count_0 = sum(1 for row in data if row["target"] == "0")
count_1 = sum(1 for row in data if row["target"] == "1")
print(f"\nTarget column created — No disease: {count_0}, Disease: {count_1}")

#  CONVERT COLUMNS TO NUMBERS

for row in data:
    row["age"]= float(row["age"])
    row["trestbps"]= float(row["trestbps"])
    row["chol"] = float(row["chol"])
    row["thalch"]= float(row["thalch"])
    row["oldpeak"]= float(row["oldpeak"])
    row["ca"]= float(row["ca"])
    row["sex"]= int(float(row["sex"]))
    row["fbs"]= int(float(row["fbs"]))
    row["exang"]= int(float(row["exang"]))
    row["target"]= int(row["target"])

print("\nColumns converted to numbers.")

# SAVE THE CLEANED DATASET
output_file = "heart_disease_cleaned.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)

print(f"\nCleaned dataset saved as {output_file}")
print("Total rows saved:", len(data))
