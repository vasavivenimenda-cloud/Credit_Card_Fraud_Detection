from src.data_loader import load_data
from src.train_model import train_and_save_model

df = load_data("dataset/creditcard.csv")

if df is not None:

    model = train_and_save_model(df)