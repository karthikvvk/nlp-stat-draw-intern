

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

country_counts = data['COUNTRY'].value_counts()

plt.figure(figsize=(10, 6))
country_counts.plot(kind='bar')
plt.xlabel('Country')
plt.ylabel('Number of Orders')
plt.title('Number of Orders by Country')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



