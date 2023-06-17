import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('fake_news_model.joblib')

# Set page configuration
st.set_page_config(
    page_title="Fake News Detection",
    page_icon=":newspaper:",
    layout="wide"
)

# Custom CSS styles
st.markdown(
    """
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #3366ff;
        text-align: center;
        padding: 20px 0;
    }
    .text-area {
        width: 80%;
        height: 150px;
        font-size: 18px;
        padding: 10px;
        border: 2px solid #3366ff;
        border-radius: 5px;
    }
    .button {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
    }
    .prediction {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def main():
    st.title("Fake News Detection")
    st.markdown("<div class='title'>Enter a news article to check if it's real or fake:</div>", unsafe_allow_html=True)
    user_input = st.text_area("Input Text")

    if st.button("Detect"):
        # Perform prediction using the loaded model
        prediction = model.predict([user_input])
        # Display the prediction
        if prediction[0] == 1:
            st.markdown("<div class='prediction' style='color:#ff3300;'>Prediction: It is fake news</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div class='prediction' style='color:#00cc44;'>Prediction: It is NOT fake news</div>",
                        unsafe_allow_html=True)

    st.markdown(
        """
        <div class='notice'>
        <p><b>Welcome to the Fake News Detection App!</b></p>
        <p>This application uses a machine learning algorithm that has been tested to have more than 99% accuracy in detecting fake news. Please note that this is an experimental project and is not intended for commercial use.</p>
        <p>By using this application, you agree that the results are for informational purposes only and at your own risk. The creators of this application are not responsible for any legal consequences or actions arising from the use of the application.</p>
        <p>Use the application responsibly to detect fake news and contribute to the fight against misinformation!</p>
        </div>
        """,
        unsafe_allow_html=True
    )
if __name__ == "__main__":
    main()
