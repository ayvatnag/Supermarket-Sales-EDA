import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r"C:\Users\VICTUS\Downloads\archive (5)\supermarket_sales - Sheet1.csv")


print(df.head())
print(df.info())
print(df.describe())
print("\nSummary statistics:")
print(df.describe())
print("Missing values:")
print(df.isnull().sum())

# 1 Descriptive Stats
mean_total = df['Total'].mean()
std_total = df['Total'].std()
print("Mean Total: ", mean_total)
print("Standard Deviation: ", std_total)

# 2 Total Sales & Average Transaction Value (ATV)
total_sales = df['Total'].sum()
total_transactions = df.shape[0]
atv = total_sales / total_transactions
print("Total Sales: ", total_sales)
print("Average Transaction Value (ATV): ", atv)

# 3 Product line vs Total Revenue
plt.figure(figsize=(10, 6))
sns.barplot(x='Product line', y='Total', data=df, estimator=np.sum, color='lightblue')
plt.title('Total Revenue by Product Line')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4 Payment Method vs Customer Type
plt.figure(figsize=(8, 5))
sns.countplot(x='Payment', hue='Customer type', data=df)
plt.title('Payment Method by Customer Type')
plt.tight_layout()
plt.show()
# 5 Correlation Heatmap
df_corr = df.drop(columns=['gross margin percentage'])
plt.figure(figsize=(10, 6))
sns.heatmap(df_corr.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (Excluding Gross Margin)')
plt.tight_layout()
plt.show()

# 6 Top Product Lines by Revenue
top_products = df.groupby('Product line')['Total'].sum().reset_index()
print("Top Product Lines by Revenue:")
print(top_products)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_products, x='Total', y='Product line', color='lightblue')
plt.title('Top Product Lines by Revenue', fontsize=14)
plt.xlabel('Revenue ($)')
plt.ylabel('Product Line')
plt.tight_layout()
plt.show()

# 7 Outlier Detection (Boxplot for Total)
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['Total'], color='lightgreen')
plt.title('Outliers in Total Amount')
plt.tight_layout()
plt.show()




