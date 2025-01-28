import fitz  # PyMuPDF
import pytesseract
import pdfplumber
import re
import io

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from adjustText import adjust_text

from transformers import pipeline, TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data
import streamlit as st

#for streamlit cloud
pytesseract.pytesseract.tesseract_cmd= '/usr/bin/tesseract'

# MDOEL TRAINING STARTS HERE
# Define labels
labels = {
    "restaurant": "FOOD", "cafe": "FOOD", "food": "FOOD", "dining": "FOOD", "swiggy": "FOOD", "faasos": "FOOD", "zomato": "FOOD",
    "mall": "SHOPPING", "store": "SHOPPING", "shopping": "SHOPPING", "retail": "SHOPPING", "amazon": "SHOPPING", "flipkart": "SHOPPING",
    "atm": "ATM", "atd": "ATM",
    "upi": "UPI", "paytm": "UPI", "funds trf": "UPI", "imps": "UPI", "rrn": "UPI", "pos": "UPI", "neft": "UPI", "rtgs": "UPI", "txn paytm": "UPI",
    "loan": "Other", "emi": "Other", "mutualfund": "Other", 
    "net txn": "Other", "cash": "Other", "interest": "Other",
    "metro": "Other", "ola": "Other", "refund": "Other", "charge": "Other", "pca": "Other"
}

# Create a mapping from string labels to integers
label_mapping = {label: idx for idx, label in enumerate(set(labels.values()))}

# Function to load model and tokenizer lazily and cache it
@st.cache_resource
def load_model_and_tokenizer(model_name="distilbert-base-uncased", num_labels=5):
    """
    Lazy load the model and tokenizer only when needed and cache them to avoid reloading.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer

# Ensure the model and tokenizer are loaded into session state
if 'model' not in st.session_state or 'tokenizer' not in st.session_state:
    st.session_state.model, st.session_state.tokenizer = load_model_and_tokenizer()

# Prepare data for training
data = []
for key, value in labels.items():
    data.append({"text": key, "label": label_mapping[value]})

df = pd.DataFrame(data)
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2)

# Function to process data and train model
def train_and_classify_model(labels, train_texts, train_labels, val_texts, val_labels):
    model_name = "distilbert-base-uncased"
    
    # Check if model is already loaded in session state
    if 'model' not in st.session_state or 'tokenizer' not in st.session_state:
        # Load the model and tokenizer into session state
        st.session_state.model, st.session_state.tokenizer = load_model_and_tokenizer(model_name, len(set(labels.values())))
    
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer

    # Tokenize the data
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    # Convert to torch dataset
    class TransactionDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = TransactionDataset(train_encodings, train_labels)
    val_dataset = TransactionDataset(val_encodings, val_labels)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained('./transaction_model')
    tokenizer.save_pretrained('./transaction_model')

    return model

# # Load the trained model for inference
# classifier = pipeline('text-classification', model='./transaction_model', tokenizer='./transaction_model')



def read_pdf(file_data):
    """
    Reads the first page of a PDF and converts it to an image.
    """
    try:
        doc = fitz.open(stream=file_data, filetype="pdf")
        page = doc.load_page(0)  # number of page
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return None

def ocr_image(image):
    """
    Extracts text from an image using Tesseract OCR.
    """
    return pytesseract.image_to_string(image, lang='eng')

def extract_transactions(text,file_data):
    """
    Parses the extracted text to find transactions and returns them as a DataFrame.
    """
    data = []
    # Wrap file_data (bytes) in a BytesIO stream
    file_stream = io.BytesIO(file_data)

    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            lines = text.split('\n')
            for line in lines:
                data.append(line.split())

    # Determine the number of columns dynamically
    max_columns = max(len(row) for row in data)
    columns = ['Transaction Date', 'Value Date', 'Description', 'Debit', 'Credit', 'Balance']
    if max_columns > len(columns):
        columns.extend([f'Extra Column {i}' for i in range(max_columns - len(columns))])            

    df = pd.DataFrame(data, columns=columns)

    print(df.columns)

    # Ensure 'Transaction Date' column exists before accessing it
    if 'Transaction Date' in df.columns:
        for i in df.index:
            if type(df["Transaction Date"][i]) == float:
                df["Transaction Date"][i] = df["Transaction Date"][i]
        df["Transaction Date"] = df["Transaction Date"]

    delete = []
    headers = ["Date", "Description", "Credit", "Debit", "Balance"]
    print(df.columns)
    for i in df.index:
        row = df.iloc[i, :].tolist()
        nan_c = 0
        # For checking empty rows
        for j in row:
            try:
                if np.isnan(j):
                    nan_c += 1
            except:
                continue
        if nan_c == len(df.columns):
            delete.append(i)

        # For checking headers in between
        for j in headers:
            if j in row:
                delete.append(i)
    df = df.drop(delete, axis=0)

    # For merging multiple lines in one
    last = 0
    delete = []
    for i in df.index:
        if type(df["Value Date"][i]) == float and type(df["Description"][i]) == str:
            buff = df["Description"][last] + df["Description"][i]
            df["Description"][last] = buff
            delete.append(i)
        else:
            last = i
    df = df.drop(delete, axis=0)

    print(df.columns)  # Debugging line to check the columns


    df["Credit"] = df["Credit"].apply(lambda x: str(x).replace(",", "") if x is not None else '0')
    df["Debit"] = df["Debit"].apply(lambda x: str(x).replace(",", "") if x is not None else '0')

    df["Credit"] = pd.to_numeric(df["Credit"].apply(lambda x: str(x).replace(",", "") if x is not None else '0'), errors='coerce').fillna(0)
    df["Debit"] = pd.to_numeric(df["Debit"].apply(lambda x: str(x).replace(",", "") if x is not None else '0'), errors='coerce').fillna(0)
    df["Value Date"] = df["Value Date"].apply(lambda x: x[3:] if x is not None else x)

    df = df[["Transaction Date", "Value Date", "Description", "Debit", "Credit", "Balance"]]
    return df


def classify_transactions(transactions):
    """
    Classifies the transactions into different categories using a pre-trained model.
    """
    t = transactions["Description"].apply(lambda x: x.lower() if x is not None else x)

    # Removing numbers and special characters
    text = t.replace(to_replace="[0-9]", value="", regex=True).apply(
        lambda x: x.replace("/", "").replace("\\", "").replace(":", "").replace("\n", " ").replace("-", " ").replace("/", " ") if x is not None else x)

    # Removing extra spaces created due to the above step
    for i in range(len(text)):
        if i >= len(text):
            print(f"Index {i} out of bounds for text with length {len(text)}")
            continue
        x = text.iloc[i].split() if text.iloc[i] is not None else []
        for j in range(len(x)):
            x[j] = x[j].strip()
        text.iloc[i] = " ".join(x)

    labs = []

    # Labelling the transaction according to the dictionary defined
    for i in text:
        f = 0
        for j in list(labels.keys()):
            if j in i:
                labs.append(labels[j])
                f = 1
                break
        if f == 0:
            labs.append("Other")
    transactions["Category"] = pd.DataFrame(labs)

    x = transactions.Description.apply(lambda x: re.findall(r'[\w\.-]+@[\w\.-]+', x) if x is not None else [])
    transactions["Remark"] = pd.DataFrame(x)

    return transactions

import matplotlib.pyplot as plt
from adjustText import adjust_text

def visualize_data(transactions):
    """
    Visualizes the classified transactions as a pie chart.
    """
    category_counts = transactions['Category'].value_counts()
    fig = plt.figure(figsize=(10, 6))

    # Define aesthetic colors for the pie chart
    colors = ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1', '#45B8AC', '#EFC050', '#5B5EA6', '#9B2335', '#BC243C']

    wedges, texts, autotexts = plt.pie(
        category_counts,
        labels=category_counts.index,
        autopct='%1.1f%%',
        labeldistance=1.1,
        colors=colors[:len(category_counts)]  # Match number of categories to colors
    )

    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.axis('equal')

    # Adjust label positions to avoid overlap
    for text in texts:
        x, y = text.get_position()
        text.set_position((1.2 * x, 1.2 * y))
    
    adjust_text(texts, force_points=0.2, force_text=0.2, expand_points=(1, 1), expand_text=(1, 1), arrowprops=dict(arrowstyle="->", color='black', lw=0.5))
    
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_bbox(dict(facecolor='white', edgecolor='none', pad=1))
    
    plt.legend(
        category_counts.index,
        title="Categories",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )

    plt.title('Transaction Classification')
    return fig


def main(file_obj):
    """
    Main function to process uploaded PDF file.
    """

    # Read PDF and process
    file_data = file_obj.read()
    image = read_pdf(file_data)
    if image is None:
        return None

    text = ocr_image(image)
    transactions = extract_transactions(text, file_data)
    transactions = classify_transactions(transactions)

    # Generate and return visualization
    fig = visualize_data(transactions)
    return fig

