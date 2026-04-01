import joblib
import pandas as pd

MODEL_PATH = "models/predict_freight_model.pkl"


def load_model(model_path: str = MODEL_PATH):
    return joblib.load(model_path)


def predict_freight_cost(quantity, dollars):

    model = load_model()

    input_df = pd.DataFrame({
        "Quantity": [quantity],
        "Dollars": [dollars]
    })

    prediction = model.predict(input_df)

    return float(prediction[0])


if __name__ == "__main__":

    quantity = 1200
    dollars = 18500

    prediction = predict_freight_cost(quantity, dollars)

    print("Predicted Freight:", prediction)