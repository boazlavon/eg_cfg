import pandas as pd
import argparse
import sys


def process_csv(input_file, output_file=None):
    df = pd.read_csv(input_file)

    # Example: Filter rows where a column value is greater than 10
    # df = df[df['column_name'] > 10]

    if output_file:
        df.to_csv(output_file, index=False)
    else:
        df.to_csv(sys.stdout, index=False)


def main():
    parser = argparse.ArgumentParser(description="Process CSV files.")
    parser.add_argument("input_file", help="Input CSV file.")
    parser.add_argument("-o", "--output_file", help="Output CSV file (optional).")
    args = parser.parse_args()

    process_csv(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
