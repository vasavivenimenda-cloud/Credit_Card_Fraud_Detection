import matplotlib.pyplot as plt


def perform_eda(df):
    """
    Perform Exploratory Data Analysis (EDA)
    """

    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)

    print(df["Class"].value_counts())

    print("\nFraud Percentage")

    fraud_percentage = (df["Class"].value_counts(normalize=True) * 100)

    print(fraud_percentage)

    # Plot Class Distribution
    plt.figure(figsize=(6,4))
    df["Class"].value_counts().plot(kind="bar")

    plt.title("Credit Card Fraud Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")

    plt.show()

    # Amount Distribution
    plt.figure(figsize=(8,5))
    plt.hist(df["Amount"], bins=50)

    plt.title("Transaction Amount Distribution")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")

    plt.show()

    # Time Distribution
    plt.figure(figsize=(8,5))
    plt.hist(df["Time"], bins=50)

    plt.title("Transaction Time Distribution")
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    plt.show()