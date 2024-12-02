import os
os.system('pip install -U -q "google-generativeai>=0.8.2" pandas==2.2.2 python-dotenv')

import google.generativeai as genai
import pandas as pd
import runpy
from dotenv import load_dotenv, dotenv_values


load_dotenv()
KEY=os.getenv('KEY')

genai.configure(api_key=KEY)
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)

chat_session = model.start_chat(
  history=[
  ]
)
csv_path = '/content/data.csv'
dataset = pd.read_csv(csv_path)
dataset_str = dataset.head(20).to_string(index=False)
user_query = input("Enter your query: ")

query = f"""
dataset insights:
{dataset.info()}
{dataset.describe(include='all')}

Dataset:

{dataset_str}

Generate Python code using matplotlib if this query ask for graphs/charts.
query: "{user_query}"
if not-asking for graphs/charts then just answer the question relevent to the dataset.

NOTE:Do not include citations or sources and decorative explanations.
Just return the code.
Full dataset available at 'csv_path = '/content/data.csv'
"""

print(query)
response = chat_session.send_message(query)

fh = open("plt.py", "w")
fo = response.text
fo = fo.replace("```python", "\n").replace("```", "\n")
fh.write(fo)
fh.close()
runpy.run_path('plt.py')


print(response)
