import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.weightstats import ztest

df = pd.read_csv(r"C:\Users\VICTUS\Downloads\archive (5)\supermarket_sales - Sheet1.csv")

print(df.head())
print(df.info())
print(df.describe())
print("\nSummary statistics:")
print(df.describe(include='all'))
print("Missing values:")
print(df.isnull().sum())

#1 Descriptive Stats
mean_total = df['Total'].mean()
std_total = df['Total'].std()
print("Mean Total: ",mean_total,"\nStandard Deviation: ",std_total)

#3 Total Sales & Average Transaction Value
total_sales = df['Total'].sum()
total_transactions = df.shape[0]
atv = total_sales / total_transactions

print("Total Sales: ",total_sales)
print("Average Transaction Value (ATV): ",atv)


#4 Product line vs Total
plt.figure(figsize=(10, 6))
sns.barplot(x='Product line', y='Total', data=df, estimator=np.sum)
plt.title('Total Revenue by Product Line')
plt.xticks(rotation=45)
plt.show()

#5 Payment Method vs Customer Type
plt.figure(figsize=(8, 5))
sns.countplot(x='Payment', hue='Customer type', data=df)
plt.title('Payment Method by Customer Type')
plt.show()

#6 Correlation Heatmap 
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

#7 Top Product Lines by Revenue
top_products = df.groupby('Product line')['Total'].sum().reset_index()
print("Top Product Lines by Revenue:")
print(top_products)
plt.figure(figsize=(10,6))
sns.barplot(data=top_products, x='Total', y='Product line', hue='Product line', palette='magma', legend=False)
plt.title('Top Product Lines by Revenue', fontsize=14)
plt.xlabel('Revenue ($)')
plt.ylabel('Product Line')
plt.tight_layout()
plt.show()

#8 Outlier Detection (Boxplot for Total)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x=df['Total'])
plt.title('Outliers in Total Amount')
plt.show()

#9 Choose two product lines to compare
product_a = 'Electronic accessories'
product_b = 'Food and beverages'
sales_a = df[df['Product line'] == product_a]['Total']
sales_b = df[df['Product line'] == product_b]['Total']

z_stat, p_val = ztest(sales_a, sales_b, alternative='two-sided')

print("Z-statistic: ",z_stat)
print("P-value: ",p_val)

alpha = 0.05
if p_val < alpha:
    print(f"Reject the null hypothesis: Average sales differ significantly between {product_a} and {product_b}.")
else:
    print(f"Fail to reject the null hypothesis: No significant difference in average sales between {product_a} and {product_b}.")


