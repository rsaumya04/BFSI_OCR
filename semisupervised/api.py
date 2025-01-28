import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sqlalchemy import create_engine
import streamlit as st


data =  {
    "AccountStatementOverAPIResponse": {
        "Data": {
            "AccountStatementReportResponseBody": {
                "statusCode": 200,
                "message": "OK",
                "requestTimeEpoch": "1694578382023",
                "requestId": "b94fe7a8-dab3-4b98-b511-ef1214c05fc5",
                "rowsCount": 6,
                "openingBalance": 0,
                "closingBalance": "",
                "accountNumber": "918020068###404",
                "currency": "INR",
                "data": [
                    {
                        "serialNumber": 1,
                        "transactionDate": "27/02/2023",
                        "pstdDate": "27/02/2023 13:50:55",
                        "transactionParticulars": "NEFT/AXISP00135816257/TA/Jay",
                        "chqNumber": " ",
                        "valueDate": "27/02/2023",
                        "amount": 1000,
                        "drcr": "DR",
                        "balance": -1000,
                        "paymentMode": "NEFT",
                        "utrNumber": " ",
                        "internalReferenceNumber": "AXISP00135816257",
                        "remittingBranch": "SECTOR 49,GURGAON [HR]",
                        "remittingBankName": "AXIS BANK",
                        "remittingAccountNumber": "918020068###404",
                        "remittingAccountName": "ILACCEE STROL PSOCESPRRS TEIVAMI LITED",
                        "remittingIFSC": "UTIB0001262",
                        "benficiaryBranch": " ",
                        "benficiaryName": " ",
                        "benficiaryAccountNumber": " ",
                        "benficiaryIFSC": " ",
                        "channel": "NEFT",
                        "timeStamp": "13:50:55",
                        "remarks": "CMS/0000230580000003",
                        "transactionCurrencyCode": "INR",
                        "entryDate": "27/02/2023 13:50:54",
                        "referenceId": "AXISP00135816257",
                        "transactionIdentificationCode": ""
                    },
                    {
                        "serialNumber": 2,
                        "transactionDate": "27/02/2023",
                        "pstdDate": "27/02/2023 13:50:55",
                        "transactionParticulars": "NEFT/AXISP00135816257/TA/Jay",
                        "chqNumber": " ",
                        "valueDate": "27/02/2023",
                        "amount": 1000,
                        "drcr": "DR",
                        "balance": -2000,
                        "paymentMode": "NEFT",
                        "utrNumber": " ",
                        "internalReferenceNumber": "AXISP00135816257",
                        "remittingBranch": "SECTOR 49,GURGAON [HR]",
                        "remittingBankName": "AXIS BANK",
                        "remittingAccountNumber": "918020068###404",
                        "remittingAccountName": "ILACCEE STROL PSOCESPRRS TEIVAMI LITED",
                        "remittingIFSC": "UTIB0001262",
                        "benficiaryBranch": " ",
                        "benficiaryName": " ",
                        "benficiaryAccountNumber": " ",
                        "benficiaryIFSC": " ",
                        "channel": "NEFT",
                        "timeStamp": "13:50:55",
                        "remarks": "CMS/0000230580000003",
                        "transactionCurrencyCode": "INR",
                        "entryDate": "27/02/2023 13:50:54",
                        "referenceId": "AXISP00135816257",
                        "transactionIdentificationCode": ""
                    },
                    {
                        "serialNumber": 3,
                        "transactionDate": "27/02/2023",
                        "pstdDate": "27/02/2023 14:49:56",
                        "transactionParticulars": "NEFT/AXISP00135816258/0127022300601/Dinesh",
                        "chqNumber": " ",
                        "valueDate": "27/02/2023",
                        "amount": 2000,
                        "drcr": "DR",
                        "balance": -4000,
                        "paymentMode": "NEFT",
                        "utrNumber": " ",
                        "internalReferenceNumber": "AXISP00135816258",
                        "remittingBranch": "SECTOR 49,GURGAON [HR]",
                        "remittingBankName": "AXIS BANK",
                        "remittingAccountNumber": "918020068###404",
                        "remittingAccountName": "ILACCEE STROL PSOCESPRRS TEIVAMI LITED",
                        "remittingIFSC": "UTIB0001262",
                        "benficiaryBranch": " ",
                        "benficiaryName": " ",
                        "benficiaryAccountNumber": " ",
                        "benficiaryIFSC": " ",
                        "channel": "NEFT",
                        "timeStamp": "14:49:56",
                        "remarks": "CMS/0000230580000004",
                        "transactionCurrencyCode": "INR",
                        "entryDate": "27/02/2023 14:49:56",
                        "referenceId": "AXISP00135816258",
                        "transactionIdentificationCode": ""
                    }
                ],
                "paging": {
                    "cursors": {
                        "next": "",
                        "previous": "MjAyMzAyMjcyMDIzMDIyNzEzNTA1NVMzMTc1MTEzMDEwMXx8fDY="
                    }
                }
            }
        },
        "Risk": {},
        "Links": {},
        "Meta": {}
    }
}

def get_transaction_data():
    """
    Extracts transaction data from the API response and returns it as a DataFrame.
    """
    transaction_data = data['AccountStatementOverAPIResponse']['Data']['AccountStatementReportResponseBody']['data']
    df = pd.DataFrame(transaction_data)  # Convert to DataFrame
    df['transactionDate'] = pd.to_datetime(df['transactionDate'], format='%d/%m/%Y')  # Convert dates
    return df

def generate_bar_chart(df):
 
    # Step 2: Bar Graph
    st.write("### Bar Graph: Transaction Amount by Date")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a colormap (example: 'viridis' colormap)
    colors = ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1'][0:df.shape[0]]

    # Correct way to use plt.bar()
    ax.bar(df['entryDate'], df['amount'], color=colors, label='Transaction Amount')

    # Customize plot
    ax.set_title("Transaction Amounts by Date", fontsize=16)
    ax.set_xlabel("Transaction Date", fontsize=12)
    ax.set_ylabel("Transaction Amount", fontsize=12)
    ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels
    ax.legend(title="Legend")

    # Display the graph in Streamlit
    return fig

def get_visualization():
    df = get_transaction_data()  # Extract data
    fig = generate_bar_chart(df)  # Generate visualization
    return df, fig
