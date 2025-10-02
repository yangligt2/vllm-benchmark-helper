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

# --- Define the Master CSV file ---
RESULTS_CSV_FILE = "benchmark_results_v2.csv"
FAILED_RUNS_FILE = "failed_runs.json"
MAX_RETRIES = 5
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
INPUT_LENS = [256, 512, 1024, 2048, 4096]
OUTPUT_LEN_RATIOS = [16, 8, 4, 2, 1]  # Corresponds to 1:1, 1:2, ..., 1:16
MAX_CONCURRENCY_VALUES = [16, 32, 64, 128, 256, 512]

# Generate the full list of configurations
BENCHMARK_CONFIGS = []

for input_len in INPUT_LENS:

    for ratio in OUTPUT_LEN_RATIOS:
        output_len = input_len // ratio
        if output_len > 1024:
            continue


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


def log_failed_run(config: Dict[str, Any]):
    """Appends a failed benchmark config to the failed_runs.json file."""
    try:
        # Read existing data if the file is not empty
        if os.path.exists(FAILED_RUNS_FILE) and os.path.getsize(FAILED_RUNS_FILE) > 0:
            with open(FAILED_RUNS_FILE, 'r') as f:
                failed_runs = json.load(f)
        else:
            failed_runs = []
        
        # Append new failed config and write back
        failed_runs.append(config)
        with open(FAILED_RUNS_FILE, 'w') as f:
            json.dump(failed_runs, f, indent=4)
            
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error writing to {FAILED_RUNS_FILE}: {e}")


def run_benchmark(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs a single benchmark using the provided config and returns the results.
    Retries on failure (completed != num_prompts).
    """
    for attempt in range(MAX_RETRIES):
        print(f"\n--- Running benchmark (Attempt {attempt + 1}/{MAX_RETRIES}) for config: "
              f"max_curr={config['max_curr']}, input_len={config['input_len']}, output_len={config['output_len']} ---")
        
        command = [
            "vllm", "bench", "serve",
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
            "--percentile-metrics", "ttft,tpot,itl,e2el",
            "--save-result"
        ]
        print(f"Executing: {' '.join(command)}")

        try:
            files_before = set(glob.glob("vllm-*.json"))
            subprocess.run(command, check=True, capture_output=True, text=True)
            files_after = set(glob.glob("vllm-*.json"))

            new_files = files_after - files_before
            if not new_files:
                print("Error: Benchmark ran, but no new result file (vllm-*.json) was found.")
                continue # Go to next retry attempt

            result_file = new_files.pop()
            print(f"--- Found result file: {result_file} ---")
            
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            # --- Success Condition Check ---
            if results.get("completed") == config.get("num_prompts"):
                print("--- Benchmark successful: completed requests match num_prompts. ---")
                # Move successful result to archive
                raw_results_dir = "raw_results"
                os.makedirs(raw_results_dir, exist_ok=True)
                archive_filepath = os.path.join(raw_results_dir, os.path.basename(result_file))
                os.rename(result_file, archive_filepath)
                print(f"--- Raw results saved to: {archive_filepath} ---")
                return results
            else:
                completed = results.get("completed", "N/A")
                num_prompts = config.get("num_prompts", "N/A")
                print(f"--- Benchmark failed: Mismatch in requests. Completed: {completed}, Expected: {num_prompts}. ---")
                os.remove(result_file) # Clean up partial result file

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error running benchmark subprocess: {e}")
            if isinstance(e, subprocess.CalledProcessError):
                print(f"Stderr: {e.stderr}")

        # If the attempt failed, wait before retrying
        if attempt < MAX_RETRIES - 1:
            print(f"Cooldown for {GPU_COOLDOWN_SEC} seconds before retry...")
            time.sleep(GPU_COOLDOWN_SEC)

    # If all retries fail
    print(f"--- Benchmark failed after {MAX_RETRIES} attempts. Logging to {FAILED_RUNS_FILE} ---")
    log_failed_run(config)
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