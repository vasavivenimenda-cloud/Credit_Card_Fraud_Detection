from src.data_loader import load_data
from src.preprocessing import dataset_info
from src.eda import perform_eda
from src.train_model import prepare_data, train_models
from src.evaluate_model import evaluate_models

df = load_data("dataset/creditcard.csv")

if df is not None:

    print(df.head())

    dataset_info(df)

    perform_eda(df)

    X_train, X_test, y_train, y_test = prepare_data(df)

    models = train_models(X_train, y_train)

    best_model = evaluate_models(models, X_test, y_test)

    print("\n✅ Best Model Selected Successfully")