import argparse
import pickle

from analyze_sae import print_analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_location", type=str, default="analysis.pickle")
    args = parser.parse_args()

    with open(args.pickle_location, "rb") as f:
        analysis_result = pickle.load(f)
        print_analysis(analysis_result)
