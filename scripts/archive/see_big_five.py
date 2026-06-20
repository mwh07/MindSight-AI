import pandas as pd

df = pd.read_csv(r"C:\Users\Lenovo\Desktop\MINDSIGHT\datasets\big_five_personality_clean.csv")

print("Shape:", df.shape)
print("\nColumns:")
for i, col in enumerate(df.columns):
    print(f"  {i}: {col}")
print("\nDtypes:\n", df.dtypes)