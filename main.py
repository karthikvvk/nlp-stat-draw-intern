import os
import google.generativeai as genai
import pandas as pd


genai.configure(api_key="AIzaSyDBsjpSUMVG0OYo5fxnM8GlrvJbEzp7MLk")
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

csv_path = 'data.csv'
dataset = pd.read_csv(csv_path)

dataset_str = dataset.head(20).to_string(index=False)  # Optional: show only a small sample of the dataset
user_query = input("enter your query: ")
# Query to generate Python code for a line graph
query = f"""
dataset insights:
{dataset.info()}
{dataset.describe(include='all')}

Dataset:

{dataset_str}

Generate Python code using matplotlib if this query ask for graphs/charts.

query: "{user_query}"

If not-asking for graphs/charts then just answer the question relevent to the dataset.

NOTE:Do not include citations or sources and decorative explanations.
Just return the code.
Full dataset available at 'data.csv'.
Include "FLAG" to indicate type of IO: "TEXT-GEMINI":textual query; "GRAPH-GEMINI":graphical query
"""

print(query)
# Get the response from the model
response = chat_session.send_message(query)

fh = open("cd.py", "w")
fo = response.text
fo = fo.replace("```python", "\n").replace("```", "\n")
fh.write(fo)
fh.close()
os.system("python cd.py")


print(response)
