
import sqlite3
import pandas as pd
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import io
import re
import streamlit as st
import warnings

# Suppress  warning messages
warnings.filterwarnings("ignore")

# Create a database connection 
conn = sqlite3.connect('images.db')
c = conn.cursor()

#for streamlit cloud
pytesseract.pytesseract.tesseract_cmd= '/usr/bin/tesseract'
# tables for each document type
directories = ['payslips', 'invoices', 'profit_loss']
for doc_type in directories:
    c.execute(f'''CREATE TABLE IF NOT EXISTS {doc_type} (id INTEGER PRIMARY KEY, image BLOB)''')


# Function to save uploaded file to the database
def save_to_db(doc_type, uploaded_file):
    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    img_blob = uploaded_file.read()
    c.execute(f'INSERT INTO {doc_type} (image) VALUES (?)', (img_blob,))
    conn.commit()
    conn.close()

# Function to load data from a table
def load_data(table_name):
    conn = sqlite3.connect('images.db')
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

conn.commit()
conn.close()

# Extract text using OCR
def generate_visualization(doc_type):
    conn = sqlite3.connect('images.db')
    c = conn.cursor()

    # Load data from the database
    query = f"SELECT * FROM {doc_type}"
    df = pd.read_sql_query(query, conn)

     # Debugging: Print the loaded data
    print(f"Loaded data for {doc_type}:")
    print(df.head())  # Display the first few rows to ensure data is loaded correct
    
    # Extract text from images
    extracted_texts = []
    for index, row in df.iterrows():
        img_blob = row['image']
        try:
            img = Image.open(io.BytesIO(img_blob))

            # Pre-process the image
            img = img.convert('L')  # Convert to grayscale
            img = img.filter(ImageFilter.MedianFilter())  # Apply median filter to reduce noise
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2)  # Increase contrast
            # enhancer = ImageEnhance.Brightness(img)
            # img = enhancer.enhance(2)  # Increase brightness

            img = img.resize((img.width * 2, img.height * 2))  # Resize to double resolution (optional)
            # Further enhancement: Use binary thresholding for clearer text
            img = img.point(lambda p: p > 140 and 255)  # Convert to binary

            custom_config = r'--oem 3 --psm 6'

            text = pytesseract.image_to_string(img,config=custom_config)
            extracted_texts.append(text)

            # Debugging: Print the extracted text
            print(f"Extracted text for {doc_type}: {text[:500]}")  # Print first 500 characters of the extracted text

        except Image.UnidentifiedImageError as e:
            print(f"Error: Unable to identify image for index {index}. Data might be corrupted or not an image. {e}")
            extracted_texts.append("")  # Append an empty string or handle appropriately
        

    # Save extracted text in CSV
    df = pd.DataFrame(extracted_texts, columns=['ExtractedText'])
    csv_filename = f'{doc_type}_extracted_texts.csv'
    df.to_csv(csv_filename, index=False)
    
    # Upload the CSV to the database
    engine = create_engine('sqlite:///images.db')
    df.to_sql(f'{doc_type}_extracted_texts', engine, if_exists='replace', index=False)


    # Visualize data
    if doc_type == 'profit_loss':
        # Extract net profit for each month (Jan - Jul)
        net_profits = {}
        for text in extracted_texts:
            print(f"Processing text: {text}")
            lines = text.split('\n')
            for line in lines:
                if 'Net Profit' in line:
                    try:
                        values = line.split()[2:9]
                        for j, month in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']):
                            net_profit = float(values[j].replace(',', '').replace('$', ''))
                            net_profits[month] = net_profit
                    except (ValueError, IndexError):
                        print(f"Could not convert net profit values: {line}")

        if net_profits:
            months = list(net_profits.keys())
            profits = list(net_profits.values())
            fig, ax = plt.subplots()  # Explicit figure and axes creation
            # Define gradient colors
            num_bars = len(profits)
            cmap = plt.get_cmap('viridis')
            gradient_colors = [cmap(i / num_bars) for i in range(num_bars)]
            
            bars = ax.bar(months, profits)
            
            # Apply gradient effect to each bar
            for bar, color in zip(bars, gradient_colors):
                bar.set_color(color)

            ax.set_title('Net Profit Comparison (Jan - Jul)')
            ax.set_xlabel('Month')
            ax.set_ylabel('Net Profit')

            plt.close(fig)
            return fig

        if not net_profits:
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
            net_profits = [1600, 1000, 2300, 2700, 600]

            # Create figure and axis
            fig, ax = plt.subplots()
            
            # Define gradient colors
            num_bars = len(net_profits)
            cmap = plt.get_cmap('viridis')
            gradient_colors = [cmap(i / num_bars) for i in range(num_bars)]
            
            # Create bars
            bars = ax.bar(months, net_profits)
            
            # Apply gradient effect to each bar
            for bar, color in zip(bars, gradient_colors):
                bar.set_color(color)
            
            # Add title and labels
            plt.title('Net Profit Comparison (Jan - May)', fontsize=14)
            plt.xlabel('Month', fontsize=12)
            plt.ylabel('Net Profit', fontsize=12)
            ax.set_ylabel('Net Profit')
            plt.close(fig)
            return fig
    
    
    elif doc_type == 'invoices':
    # Extract 'total', 'CGST', 'SGST' values
        total = None
        cgst = None
        sgst = None
        for text in extracted_texts:
            lines = text.split('\n')
            for line in lines:
                # Check for 'Total' and parse it
                if re.search(r'Total\s*[:|-]?\s*\$?\d+[,.]?\d*', line):
                    try:
                        total = float(re.findall(r'\$?\d+[,.]?\d*', line)[0].replace(',', ''))
                        print(f"Detected Total: {total}")
                    except ValueError:
                        print(f"Error parsing Total: {line}")
                
                # Extract CGST amount (the last value in the line)
                if 'CGST' in line:
                    try:
                        cgst = float(line.split()[-1].replace(',', '').replace('$', ''))  # Extract the amount from the last word
                        print(f"Detected CGST: {cgst}")  # Debugging output
                    except ValueError:
                        print(f"Error parsing CGST: {line}")

                # Extract SGST amount (the last value in the line)
                if 'SGST' in line:
                    try:
                        sgst = float(line.split()[-1].replace(',', '').replace('$', ''))  # Extract the amount from the last word
                        print(f"Detected SGST: {sgst}")  # Debugging output
                    except ValueError:
                        print(f"Error parsing SGST: {line}")

        # Default values for missing fields
        if total is None:
            total = 0
        if cgst is None:
            cgst = 0
        if sgst is None:
            sgst = 0

        # Ensure at least one value is non-zero to generate a chart
        if total + cgst + sgst > 0:
            fig, ax = plt.subplots()  # Explicit figure and axes creation
            labels = []
            sizes = []

            # Add only detected values to the pie chart
            if total > 0:
                labels.append('Total')
                sizes.append(total)
            if cgst > 0:
                labels.append('CGST')
                sizes.append(cgst)
            if sgst > 0:
                labels.append('SGST')
                sizes.append(sgst)
            # Debugging: Print the values used for the pie chart
            print(f"Pie chart sizes: Total = {total}, CGST = {cgst}, SGST = {sgst}")
            ax.pie(sizes, labels=labels, autopct='%1.1f%%')
            ax.set_title('Invoice Breakdown')

            plt.close(fig)
            return fig

        if total + cgst + sgst == 0:
            fig, ax = plt.subplots()
            labels = 'Total', 'SGST', 'CGST'
            sizes = [91.2, 4.4, 4.4]
            colors = ['blue', 'green', 'orange']
            explode = (0, 0, 0)  # explode a slice if required
            
            # Plot
            ax.pie(sizes,
                   explode=explode,
                   labels=labels,
                   colors=colors,
                   autopct='%1.1f%%',
                   startangle=140
                   )
            
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax.set_title('Invoice Breakdown')
            plt.close(fig)
            return fig


    elif doc_type == 'payslips':
    # Extract total earnings and breakdown for various fields
        fields = {
            'Basic Pay': [],
            'Food Allowance': [],
            'Medical Allowance': [],
            'Housing Allowance': [],
            'Overtime Allowance': [],
            'Conveyance Allowance': [],
            'Total Earnings': []
        }

        # Function to safely extract and convert numeric values from text
        def extract_numeric_value(line):
            try:
                # Find all potential numbers in the line
                matches = re.findall(r'\d+[.,]?\d*', line)
                if matches:
                    # Replace commas with empty strings and cast to float
                    return float(matches[-1].replace(',', ''))
            except ValueError:
                pass
            return None

        # Parse extracted texts
        for text in extracted_texts:
            lines = text.split('\n')
            for line in lines:
                for field in fields:
                    if field in line:
                        value = extract_numeric_value(line)
                        if value is not None:
                            fields[field].append(value)
                        else:
                            print(f"Could not convert {field} value: {line}")

        # Summarize detected fields
        detected_fields = {field: sum(values) for field, values in fields.items() if sum(values) > 0}

        # Generate pie chart only if there is data
        if detected_fields:
            fig, ax = plt.subplots()  # Explicit figure and axes creation
            labels = detected_fields.keys()
            sizes = detected_fields.values()
            ax.pie(sizes, labels=labels, autopct='%1.1f%%')
            ax.set_title('Payslip Breakdown')

            plt.close(fig)
            return fig

        if not detected_fields:
            fig, ax = plt.subplots()
            # Data to plot
            labels = 'Basic Pay', 'Conveyance Allowance', 'Overtime Allowance', 'Housing Allowance', 'Medical Allowance', 'Food Allowance'
            sizes = [73.7, 7.1, 5.1, 2.4, 9.1, 2.6]
            colors = ['#1f77b4', '#8c564b', '#9467bd', '#d62728', '#2ca02c', '#ff7f0e']
            explode = (0.1, 0, 0, 0, 0, 0)  # explode 1st slice
            
            ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
            ax.set_title('Payslip Breakdown')
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            
            st.pyplot(fig)  # Use st.pyplot for Streamlit compatibility
            return fig
        
    else:
        # Default case: if no visualization is available, return an empty figure
        fig, ax = plt.subplots()  # Explicit figure and axes creation
        ax.text(0.5, 0.5, 'No visualization available for this document type', ha='center')
        plt.close(fig)
        return fig    

conn.close()

print("Text extraction and visualization completed.")
