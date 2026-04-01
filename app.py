import streamlit as st

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag


# Page configuration
st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📊",
    layout="wide"
)

# Header
st.title("Vendor Invoice Intelligence Portal")

st.markdown("""
### AI-Driven Freight Cost Prediction & Invoice Risk Flagging

This portal helps finance teams:

• Predict freight costs  
• Detect risky invoices  
• Reduce manual approvals
""")

st.divider()


# Sidebar
st.sidebar.title("Model Selection")

selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    ("Freight Cost Prediction", "Invoice Manual Approval Flag")
)


# -----------------------------------------------------
# Freight Cost Prediction
# -----------------------------------------------------
if selected_model == "Freight Cost Prediction":

    st.subheader("Freight Cost Prediction")

    st.markdown("""
Predict freight cost using **Quantity** and **Invoice Dollars**
to support budgeting and vendor negotiations.
""")

    with st.form("freight_form"):

        col1, col2 = st.columns(2)

        with col1:
            quantity = st.number_input(
                "Quantity",
                min_value=1,
                value=1200
            )

        with col2:
            dollars = st.number_input(
                "Invoice Dollars",
                min_value=1.0,
                value=18500.0
            )

        submit_freight = st.form_submit_button("Predict Freight Cost")

    if submit_freight:

        prediction = predict_freight_cost(quantity, dollars)

        st.success("Prediction completed successfully")

        st.metric(
            label="Estimated Freight Cost",
            value=f"${prediction:,.2f}"
        )


# -----------------------------------------------------
# Invoice Flag Prediction
# -----------------------------------------------------
if selected_model == "Invoice Manual Approval Flag":

    st.subheader("Invoice Manual Approval Prediction")

    with st.form("invoice_flag_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            invoice_quantity = st.number_input(
                "Invoice Quantity",
                min_value=1,
                value=50
            )

            freight = st.number_input(
                "Freight Cost",
                min_value=0.0,
                value=1.73
            )

        with col2:

            invoice_dollars = st.number_input(
                "Invoice Dollars",
                min_value=1.0,
                value=352.95
            )

            total_item_quantity = st.number_input(
                "Total Item Quantity",
                min_value=1,
                value=162
            )

        with col3:

            total_item_dollars = st.number_input(
                "Total Item Dollars",
                min_value=1.0,
                value=2476.0
            )

        submit_flag = st.form_submit_button("Predict Invoice Risk")

    if submit_flag:

        input_data = {
            "invoice_quantity": [invoice_quantity],
            "invoice_dollars": [invoice_dollars],
            "Freight": [freight],
            "total_item_quantity": [total_item_quantity],
            "total_item_dollars": [total_item_dollars]
        }

        flag_prediction = predict_invoice_flag(input_data)["Predicted_Flag"]

        is_flagged = bool(flag_prediction[0])

        if is_flagged:
            st.error("Invoice requires MANUAL APPROVAL")
        else:
            st.success("Invoice is SAFE for Auto-Approval")