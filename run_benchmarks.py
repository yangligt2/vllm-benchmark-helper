import os
import subprocess
import sys
import glob
import json
import csv
import time
from datetime import datetime
from typing import Dict, Any, List

# --- Server details (adjust if needed) ---
IP = "34.0.139.51"
PORT = 80
BENCHMARK_DIR = "/usr/local/google/home/yangligt/workplaces/vllm/benchmarks"

# --- Define the Master CSV file ---
RESULTS_CSV_FILE = "benchmark_results.csv"
GPU_COOLDOWN_SEC = 60

# --- List of all benchmark configurations to run ---
# Each dictionary represents one benchmark run.
BENCHMARK_CONFIGS = [
]

# --- Dynamic Benchmark Configuration ---

# Base settings shared by all benchmark runs
BASE_CONFIG = {
    "model": "nvidia/Llama-3.3-70B-Instruct-FP8",
    "tokenizer": "nvidia/Llama-3.3-70B-Instruct-FP8",
    "hardware": "2x_a3ultra_8xh200",
    "pd_enabled": "true",
    "prefill_node": 1,
    "prefill_dp": 4,
    "prefill_tp": 2,
    "decode_node": 1,
    "decode_dp": 1,
    "decode_tp": 8,    
    "req_rate": "inf", # Using "inf" for throughput benchmarks
}

# --- Define the parameter spaces for the 3-D sweep ---
INPUT_LENS = [4096, 2048, 1024, 512, 256]
OUTPUT_LEN_RATIOS = [2, 4, 8, 16]  # Corresponds to 1:1, 1:2, ..., 1:16
MAX_CONCURRENCY_VALUES = [16, 32, 64, 128, 256, 512]

# Generate the full list of configurations
BENCHMARK_CONFIGS = []

for input_len in INPUT_LENS:

    for ratio in OUTPUT_LEN_RATIOS:
        output_len = input_len // ratio

        for max_curr in MAX_CONCURRENCY_VALUES:
            config = BASE_CONFIG.copy()
            num_prompts = max(256, 4 * max_curr)

            config.update({
                "input_len": input_len,
                "output_len": output_len,
                "num_prompts": num_prompts,
                "max_curr": max_curr,
            })
            BENCHMARK_CONFIGS.append(config)



def run_benchmark(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs a single benchmark using the provided config and returns the results.
    """
    temp_result_file = "temp_benchmark_output.json"
    
    # Construct the command
    command = [
        sys.executable, f"{BENCHMARK_DIR}/benchmark_serving.py",
        "--base-url", f"http://{IP}:{PORT}",
        "--backend", "vllm",
        "--model", config["model"],
        "--endpoint", "/v1/completions",
        "--tokenizer", config["tokenizer"],
        "--dataset-name", "random",
        "--random-input-len", str(config["input_len"]),
        "--random-output-len", str(config["output_len"]),
        "--num-prompts", str(config["num_prompts"]),
        "--request-rate", str(config["req_rate"]),
        "--max-concurrency", str(config["max_curr"]),
        "--save-result"  # The script generates its own output file
    ]
    
    print(f"\n--- Running benchmark for config: {config['max_curr']} max_curr, {config['input_len']} input_len ---")
    print(f"Executing: {' '.join(command)}")
    
    try:
        # Run the benchmark command
        # Find existing result files before the run to isolate the new one.
        files_before = set(glob.glob("vllm-*.json"))
        subprocess.run(command, check=True)
        files_after = set(glob.glob("vllm-*.json"))

        new_files = files_after - files_before
        if not new_files:
            print("Error: Benchmark ran, but no new result file (vllm-*.json) was found.")
            return None
        if len(new_files) > 1:
            print(f"Warning: Multiple new result files found. Using the first one: {list(new_files)[0]}")

        result_file = new_files.pop()
        print(f"--- Found result file: {result_file} ---")
        
        # Load the results from the JSON file
        with open(result_file, 'r') as f:
            results = json.load(f)
            
        # --- Move the raw JSON result to the archive directory ---
        raw_results_dir = "raw_results"
        os.makedirs(raw_results_dir, exist_ok=True)

        # Move the original file to the raw_results directory
        archive_filepath = os.path.join(raw_results_dir, os.path.basename(result_file))
        os.rename(result_file, archive_filepath)
        print(f"--- Raw results saved to: {archive_filepath} ---")

        return results
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error running benchmark subprocess: {e}")
        return None

def main():
    """
    Main function to loop through configs and save results.
    """
    # Check if the CSV file needs a header.
    # This is true if the file doesn't exist or is empty.
    write_header = not os.path.exists(RESULTS_CSV_FILE) or os.path.getsize(RESULTS_CSV_FILE) == 0
    
    with open(RESULTS_CSV_FILE, 'a', newline='') as csvfile:
        writer = None

        for i, config in enumerate(BENCHMARK_CONFIGS):
            results = run_benchmark(config)
            
            if not results:
                print(f"--- Skipping results for run {i+1}/{len(BENCHMARK_CONFIGS)} due to error ---")
                continue

            # Combine the config and the benchmark results
            combined_data = config.copy()
            combined_data.update(results)
            
            # Add a timestamp for uniqueness
            combined_data["timestamp"] = datetime.now().isoformat()

            # On the first successful run, setup the CSV writer and write the header
            if writer is None:
                # Define column order: config keys first, then result keys, then timestamp.
                config_keys = list(config.keys())
                result_keys = list(results.keys())
                # The 'timestamp' key is added manually, so we add it to the fieldnames.
                fieldnames = config_keys + result_keys + ["timestamp"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                    write_header = False # Prevent writing header again in this session
            
            # Ensure all fields are present for this row
            row_to_write = {field: combined_data.get(field) for field in writer.fieldnames}
            writer.writerow(row_to_write)
            csvfile.flush() # Save progress immediately
            print(f"--- Successfully saved results for run {i+1}/{len(BENCHMARK_CONFIGS)} ---")
            print(f"GPU cooldown for {GPU_COOLDOWN_SEC} seconds...")
            time.sleep(GPU_COOLDOWN_SEC)


if __name__ == "__main__":
    main()