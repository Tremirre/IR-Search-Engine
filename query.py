import pandas as pd


def read_file(filename: str) -> pd.DataFrame:
    if filename.endswith(".pq"):
        return pd.read_parquet(filename)
    elif filename.endswith(".csv"):
        return pd.read_csv(filename)
    else:
        raise ValueError("Invalid file extension! Must be .pq or .csv")


def main():
    df = read_file("out.pq")
    print(df.head())


if __name__ == "__main__":
    main()
