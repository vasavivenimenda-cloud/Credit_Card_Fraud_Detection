from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def prepare_data(df):

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain Shape :", X_train.shape)
    print("Test Shape :", X_test.shape)

    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):

    models = {

        "Logistic Regression": LogisticRegression(max_iter=1000),

        "Decision Tree": DecisionTreeClassifier(random_state=42),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    }

    trained_models = {}

    print("\nTraining Models...\n")

    for name, model in models.items():

        model.fit(X_train, y_train)

        trained_models[name] = model

        print(f"{name} Trained Successfully")

    return trained_models