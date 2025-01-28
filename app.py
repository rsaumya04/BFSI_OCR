import streamlit as st

# Set page configuration with a custom title and layout
st.set_page_config(page_title="Finance Analyzer", layout="centered")

import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import io
from supervised.financial_statements import save_to_db, load_data, generate_visualization
from unsupervised.bs_llm import main as analyze_bank_statement
from semisupervised.api import get_visualization


# Custom CSS for background image, fonts, and styling
st.markdown(

    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap');
    </style>


    <style>
    /* Background image for home page */
    .stApp {
        background: url('https://techbullion.com/wp-content/uploads/2023/04/Best-Forex-Trading-Journals-2023-Four-of-The-Best-Platforms-Out-There.jpg') no-repeat center center fixed;
        background-size: cover;
        filter: brightness(90%); /* Dull the background image */
    }
    /* Style for the background box of the title */
    .title-box {
        background-color: rgba(255, 255, 255, 0.5); /* White background with 80% opacity */
        padding: 0px 10px; /* Providing minimal padding to ensure the box fits closely */
        border-radius: 10px; /* Round corners of the box */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Optional shadow for better visibility */
        text-align: center; /* Center-align the title text */
        margin: 0px; /* Removing any additional space from the top */
        display: inline-block; /* Ensure the box fits the title content */
        line-height: 1; /* Ensure the line height is 1 */
        height: auto; /* Ensure the height wraps tightly around the content */
        font-size: 60px; /* Matching the font size with the title */
    }
    .poppins-black-italic {
    font-family: "Poppins", serif;
    font-weight: 900;
    font-style: italic;
    }


    /* Style for the app title */
    .app-title {
        font-size: 60px; /* Larger font size */
        display: flex;
        margin-bottom: 12px;
        justify-content: center;
        align-item: center;
        font-weight: 800; /* Make the title bold */
        # font-family: 'Cambria', serif; /* Use a serif font */
        color: black !important; /* Ensure the title is black */
        text-align: center; /* Center the title */
        margin-top: 20px; /* Space from the top */
    }

    /* Disclaimer styling */
    .disclaimer {
        font-size: 14px;
        text-align: center;
        color: black; /* Jet black font color */
        margin-top: 120px;
        background-color: rgba(255, 255, 255, 0.5); /* White background with 80% opacity */
        padding: 20px; /* Add padding around the text */
        border-radius: 10px; /* Round corners of the box */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Optional shadow for better visibility */
    }
    /* Style for the note inside an opaque box */
    .note-box {
        background-color: rgba(255, 255, 255, 0.8); /* White background with 80% opacity */
        padding: 20px; /* Add padding around the text */
        border-radius: 10px; /* Rounded corners */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Optional shadow for better visibility */
        text-align: left; /* Left-align the note text */
        margin-top: 30px; /* Space between elements */
    }
     /* Styling the text inside the note box */
    .note-box p, .note-box li, .note-box h3 {
        color: black !important; /* Ensure the text is black */
    }

    /* Ensure that the list items (<li>) are black */
    .note-box ul {
        list-style-type: disc; /* Add disc bullet points */
        padding-left: 20px; /* Indentation for the list */
    }

    /* Styling the heading */
    .note-box h3 {
        font-size: 24px; /* Larger font size for the heading */
    }
    /* Target st.subheader component specifically */
    .stMarkdown h2, .stSubheader {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<div class="title-box"><div class="app-title">Finance Analyzer 📈</div></div>', unsafe_allow_html=True)

# Add spinner when app is loading
with st.spinner("Loading application, please wait..."):
    st.session_state.get("initialized", True)

# Check if the page has already been selected
if 'page' not in st.session_state:
    st.session_state['page'] = None

# Home page (landing page with buttons)
if st.session_state['page'] is None:
    st.subheader("Choose which operation you want to perform:")

    # Create buttons for options instead of selectbox
    if st.button('Analyze Financial Statements and Slips'):
        st.session_state['page'] = "Analyze Financial Statements and Slips"
        st.rerun()
    elif st.button('Analyze Expense Behavior from Bank Statements'):
        st.session_state['page'] = "Analyze Expense Behavior from Bank Statements"
        st.rerun()
    elif st.button('Retrieve and Analyze Data via Bank API'):
        st.session_state['page'] = "Retrieve and Analyze Data via Bank API"
        st.rerun()

    # Disclaimer at the bottom of the home page
    st.markdown(
        """
        <div class='disclaimer'>
        This application has been developed using open-source tools and models, including Tesseract OCR, BERT, and BART. 
        While these technologies offer significant capabilities, their results may not always be fully accurate due to various limitations inherent in the models and tools used. 
        While we strive for accuracy, results may vary based on the input data quality. Your feedback is invaluable for further improvement.
        </div>
        """,
        unsafe_allow_html=True
    )

# Option 1: Analyze Financial Statements and Slips
elif st.session_state['page'] == "Analyze Financial Statements and Slips":
    st.subheader("Analyze Financial Statements and Slips")

    # Add a back button to go back to the home page
    if st.button("Back to Home"):
        st.session_state['page'] = None
        st.rerun()

    # Function to clear database tables
    def clear_tables():
        conn = sqlite3.connect('images.db')
        c = conn.cursor()
        tables = ['payslips', 'invoices', 'profit_loss']
        for table in tables:
            c.execute(f"DELETE FROM {table}")
            conn.commit()

        conn.close()

    # Initialize session state for doc type and uploaded file
    if 'doc_type' not in st.session_state:
        st.session_state['doc_type'] = None
    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = None

    # Sidebar for selecting document type with theme colors
    st.sidebar.markdown(
        """
        <style>
        .sidebar .block-container {
            background-color: #7fdc7f;
            color: black;
            padding: 10px;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    doc_type = st.sidebar.selectbox('Select Document Type', ['profit_loss','payslips', 'invoices',])

    # Reset state when document type changes
    if doc_type != st.session_state['doc_type']:
        st.session_state['doc_type'] = doc_type
        st.session_state['uploaded_files'] = None  # Reset file upload
        clear_tables()  # Clear any previous data from the database

    # File uploader
    uploaded_files = st.file_uploader("Upload files", type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)

    # Update uploaded_files in session state
    if uploaded_files:
        st.session_state['uploaded_files'] = uploaded_files

    # Initialize data variable
    data = None

    # Process uploaded files only if they exist in session state
    if st.session_state['uploaded_files']:
        save_to_db(doc_type, st.session_state['uploaded_files'])
        st.success("Files uploaded successfully!")

        # Fetch and display data after uploading
        data = load_data(doc_type)
    else:
        st.warning("No files uploaded. Please upload files to view data.")

    # Generate and display visualization
    if data is not None and not data.empty:
        fig = generate_visualization(doc_type)
        
        st.pyplot(fig)

        # Provide download option for the visualization
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        st.download_button("Download Visualization", buf, file_name=f"{doc_type}_visualization.png", mime="image/png")
    else:
        st.info("No data available for visualization.")

# Option 2: Bank Statement Analysis
elif st.session_state['page'] == "Analyze Expense Behavior from Bank Statements":
    st.subheader("Analyze Expense Behavior from Bank Statements")

    # Add a back button to go back to the home page
    if st.button("Back to Home"):
        st.session_state['page'] = None
        st.rerun()

    # File uploader for PDFs
    uploaded_file = st.file_uploader("Upload a Bank Statement (PDF)", type=["pdf"])

    if uploaded_file:
        st.info("Processing uploaded file...")
        fig = analyze_bank_statement(uploaded_file)
        if fig:
            st.pyplot(fig)

            # Download button for visualization
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            st.download_button("Download Visualization", buf, file_name="bank_statement_visualization.png", mime="image/png")
        else:
            st.error("Failed to process the file. Please try again.")

# Option 3: Bank API Analysis
elif st.session_state['page'] == "Retrieve and Analyze Data via Bank API":
    st.subheader("Retrieve and Analyze Data via Bank API")

    # Add a back button to go back to the home page
    if st.button("Back to Home"):
        st.session_state['page'] = None
        st.rerun()

    # Heading and Note
    st.markdown(
    '''
    <div class="note-box">
        <h3>This retrieves Axis Bank Data</h3>
        <p>Data recovery via bank API is an official procedure that requires secure authentication and authorization
        from the bank's system. Typically, the data retrieval process is challenging because:</p>
        <ul>
            <li>Banks require stringent verification mechanisms to prevent unauthorized access.</li>
            <li>The API has limited access and strict rate limiting to avoid misuse.</li>
            <li>The data must be processed and filtered according to privacy regulations, such as GDPR.</li>
            <li>It requires a secure network connection to avoid man-in-the-middle attacks.</li>
            <li>Each API interaction must be logged and reported as per compliance rules.</li>
        </ul>
        <p>Hence, what you see here is just a sample of data retrieved from the Axis Bank API to demonstrate the
        transaction behavior analysis.</p>
    </div>
    ''',
    unsafe_allow_html=True
)

    # Fetch the extracted data and the generated graph from the backend
    df, fig = get_visualization()

    # Display the extracted DataFrame
    st.write("### Extracted Data from Bank API:")
    st.dataframe(df)

    # Display the bar graph
    st.write("### Bar Graph: Transaction Amount by Date")
    st.pyplot(fig)

    # Optionally add a download button for the graph
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    st.download_button("Download Visualization", buf, file_name="bank_statement_visualization.png", mime="image/png")
