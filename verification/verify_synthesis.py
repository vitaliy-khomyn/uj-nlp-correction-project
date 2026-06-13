import sys
import os

sys.path.append(os.path.abspath('.'))
from src.data.synthesis.synthesize_data import main

if __name__ == '__main__':
    print("Running refactored synthesize_data pipeline on a debug slice of 10 sentences...")
    try:
        main(
            corpus_path='data/downloaded/Wikipedia-pl-train.parquet',
            output_path='data/synthesized/test_debug.parquet',
            max_pairs=10,
            max_injections_per_word=2
        )
        print("SUCCESS: Data synthesis completed with no errors!")
        if os.path.exists('data/synthesized/test_debug.parquet'):
            os.remove('data/synthesized/test_debug.parquet')
            print("Cleaned up temporary debug output.")
    except Exception as e:
        import traceback
        print("FAILED with exception:")
        traceback.print_exc()
