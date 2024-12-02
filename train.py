import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import csv

# Open your CSV file
with open('train_data.csv', mode='r') as file:
    reader = csv.DictReader(file)  # Read the CSV
    data = []  # Initialize empty list to hold the output

    # Loop through each row in the CSV file
    for row in reader:
        query = row['query']
        keywords = row['keywords']
        graph_requirement = row['graph_requirement']
        
        # Append the formatted dictionary to the list
        data.append({
            "query": query,
            "keywords": keywords,
            "graph_requirement": graph_requirement
        })




formatted_data = {
    "query": [item["query"] for item in data],
    "keywords": [item["keywords"] for item in data],
    "graph_requirement": [item["graph_requirement"] for item in data],
}
# Convert to Hugging Face Dataset
dataset = Dataset.from_dict(formatted_data)

# Tokenizer and encode the data
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize_function(examples):
    return tokenizer(examples["query"], padding=True, truncation=True, max_length=512)

# Tokenize the data
encoded_dataset = dataset.map(tokenize_function, batched=True)

# Label encoding for graph_requirement
# Label encoding for graph_requirement
encoded_dataset = encoded_dataset.map(
    lambda x: {"labels": [1 if label == "needed" else 0 for label in x["graph_requirement"]]},
    batched=True
)


# Split the dataset into train and test sets
train_dataset, test_dataset = encoded_dataset.train_test_split(test_size=0.2).values()

# Model definition
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./trained_roberta")
tokenizer.save_pretrained("./trained_roberta")
