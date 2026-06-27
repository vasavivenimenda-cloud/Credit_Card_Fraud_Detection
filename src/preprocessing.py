def dataset_info(df):
    """
    Display basic information about the dataset.
    """

    print("\n" + "=" * 60)
    print("DATASET SHAPE")
    print("=" * 60)
    print(df.shape)

    print("\n" + "=" * 60)
    print("COLUMN NAMES")
    print("=" * 60)
    print(df.columns.tolist())

    print("\n" + "=" * 60)
    print("DATA TYPES")
    print("=" * 60)
    print(df.dtypes)

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    print(df.isnull().sum())

    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY")
    print("=" * 60)
    print(df.describe())