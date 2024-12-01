

import pandas as pd
import matplotlib.pyplot as plt

csv_path = '/content/sample_data/data.csv'
df = pd.read_csv(csv_path)

average_sales_per_product_line = df.groupby('PRODUCTLINE')['SALES'].mean()

plt.figure(figsize=(10, 6))
average_sales_per_product_line.plot(kind='bar')
plt.xlabel('Product Line')
plt.ylabel('Average Sales')
plt.title('Average Sales per Product Line')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



