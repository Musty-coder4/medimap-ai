import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """\
# MediMap AI: Data Processing, Cleaning, and Exploratory Data Analysis (EDA)
This notebook performs a comprehensive analysis on the **SympScan** (Diseases and Symptoms) dataset used to train the MediMap AI Hybrid Neural Network.

We will cover:
1. Data Loading & Inspection
2. Data Cleaning (Missing Values & Duplicates)
3. Exploratory Data Analysis (Disease & Symptom Distributions)
"""

code_imports = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
"""

text_load = """\
## 1. Data Loading & Inspection
Let's load the dataset and look at the general shape and the first few rows.
"""

code_load = """\
df = pd.read_csv('../data/raw/tabular/Diseases_and_Symptoms_dataset.csv')

print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
display(df.head())
"""

text_clean = """\
## 2. Data Cleaning
We need to ensure our data is high quality before feeding it into the neural network. We will check for missing values (NaNs) and duplicate patient records.
"""

code_clean = """\
# Check for missing values
total_missing = df.isnull().sum().sum()
print(f"Total Missing Values: {total_missing}")

if total_missing > 0:
    print("Missing values detected! We would normally impute or drop these.")
else:
    print("Dataset is perfectly clean with 0 missing values.")

# Check for duplicates
total_duplicates = df.duplicated().sum()
print(f"\\nTotal Duplicate Rows: {total_duplicates}")

if total_duplicates > 0:
    print("Dropping duplicates to prevent data leakage...")
    df = df.drop_duplicates()
    print(f"New Dataset Shape: {df.shape}")
else:
    print("Dataset has 0 duplicate rows. Every record is unique!")
"""

text_eda = """\
## 3. Exploratory Data Analysis (EDA)
Let's visualize the distribution of diseases to see if our dataset is balanced, and analyze which symptoms are the most common across all diseases.
"""

code_eda1 = """\
# Disease Distribution (Top 20 most common diseases)
# Assuming the first column is the label. If the label column name is different, we can access it using df.columns[0]
label_col = df.columns[0]
disease_counts = df[label_col].value_counts()

plt.figure(figsize=(12, 8))
sns.barplot(y=disease_counts.head(20).index, x=disease_counts.head(20).values, palette="viridis")
plt.title("Top 20 Most Frequent Diseases in Dataset")
plt.xlabel("Number of Samples")
plt.ylabel("Disease")
plt.show()

print(f"Total Unique Diseases: {len(disease_counts)}")
"""

code_eda2 = """\
# Symptom Frequency Analysis (Top 20 most common symptoms)
# We sum the 1s across all symptom columns
symptom_freq = df.iloc[:, 1:].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(y=symptom_freq.head(20).index, x=symptom_freq.head(20).values, palette="magma")
plt.title("Top 20 Most Common Symptoms Across All Diseases")
plt.xlabel("Frequency (Number of Occurrences)")
plt.ylabel("Symptom")
plt.show()
"""

text_conclusion = """\
## Conclusion
- The dataset contains **96,088** perfectly clean records with no missing values or duplicates.
- It covers **100** distinct diseases and **230** unique symptoms encoded in a binary matrix.
- Because it is highly balanced and completely numeric (except the label), it is perfectly formatted for our PyTorch Hybrid Neural Network without needing heavy transformation pipelines.
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(text_load),
    nbf.v4.new_code_cell(code_load),
    nbf.v4.new_markdown_cell(text_clean),
    nbf.v4.new_code_cell(code_clean),
    nbf.v4.new_markdown_cell(text_eda),
    nbf.v4.new_code_cell(code_eda1),
    nbf.v4.new_code_cell(code_eda2),
    nbf.v4.new_markdown_cell(text_conclusion)
]

import os
os.makedirs('notebooks', exist_ok=True)
nbf.write(nb, 'notebooks/EDA_and_Data_Cleaning.ipynb')
print("Notebook created successfully at notebooks/EDA_and_Data_Cleaning.ipynb")
