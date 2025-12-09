import os
import subprocess
import sys
import glob
import json
import csv
import time
import argparse
import yaml
from datetime import datetime
from typing import Dict, Any, List

def generate_benchmark_configs(base_config: Dict[str, Any], sweep_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generates a list of benchmark configurations from sweep parameters."""
    
    configs = []
    req_rates = sweep_params.get("req_rates", [])
    input_lens = sweep_params.get("input_lens", [])
    output_len_ratios = sweep_params.get("input_to_output_len_ratios", [])
    max_concurrency_values = sweep_params.get("max_concurrency_values", [])

    for req_rate in req_rates:
        for input_len in input_lens:
            for ratio in output_len_ratios:
                output_len = int(round(input_len / ratio))
                if output_len > 1024:
                    continue

                for max_curr in max_concurrency_values:
                    # Logic to pair req_rate and max_concurrency correctly:
                    # - If req_rate is 'inf' (throughput), max_curr must be a number.
                    # - If req_rate is a number (latency), max_curr should be None.
                    if req_rate == "inf" and max_curr is None:
                        continue
                    if req_rate != "inf" and max_curr is not None:
                        continue

                    config = base_config.copy()

                    # Check if num_prompts is explicitly provided in the base_config
                    explicit_num_prompts = config.get("num_prompts")

                    num_prompts_for_this_config = None
                    if isinstance(explicit_num_prompts, (int, float)):
                        num_prompts_for_this_config = int(explicit_num_prompts)

                    if num_prompts_for_this_config is None:
                        # Calculate num_prompts if not explicitly provided or not a valid number
                        calculated_num_prompts_base = 0
                        if max_curr is not None:
                            # Throughput test (req_rate is "inf", max_curr is a number)
                            calculated_num_prompts_base = 10 * max_curr
                        else:
                            # Latency test (req_rate is a number, max_curr is None)
                            # Based on the pairing logic, req_rate must be a number here.
                            try:
                                req_rate_float = float(req_rate)
                                calculated_num_prompts_base = int(req_rate_float * 60) # req_rate * 60 seconds
                            except ValueError:
                                # This case should ideally not be reached due to earlier pairing logic.
                                # If it is, it means req_rate is not a number while max_curr is None,
                                # which is an invalid state that should have been skipped.
                                print(f"Warning: Unexpected state - req_rate '{req_rate}' is not a number while max_curr is None. Defaulting num_prompts.")
                                calculated_num_prompts_base = 512 # Fallback

                        num_prompts_for_this_config = max(512, calculated_num_prompts_base)

                    config.update({
                        "req_rate": req_rate,
                        "input_len": input_len,
                        "output_len": output_len, # This line was missing in the original selection, but is part of the config update.
                        "num_prompts": num_prompts_for_this_config,
                        "max_curr": max_curr,
                    })
                    configs.append(config)
    return configs


def log_failed_run(config: Dict[str, Any], failed_runs_file: str):
    """Appends a failed benchmark config to the specified JSON file."""
    try:
        # Read existing data if the file is not empty
        if os.path.exists(failed_runs_file) and os.path.getsize(failed_runs_file) > 0:
            with open(failed_runs_file, 'r') as f:
                failed_runs = json.load(f)
        else:
            failed_runs = []
        
        # Append new failed config and write back
        failed_runs.append(config)
        with open(failed_runs_file, 'w') as f:
            json.dump(failed_runs, f, indent=4)
            
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error writing to {failed_runs_file}: {e}")


def run_benchmark(config: Dict[str, Any], exp_setup: Dict[str, Any], raw_results_dir: str, failed_runs_file: str) -> Dict[str, Any]:
    """
    Runs a single benchmark using the provided config and returns the results.
    Retries on failure (completed != num_prompts).
    """
    max_retries = exp_setup.get("max_retries", 3)
    gpu_cooldown_sec = exp_setup.get("gpu_cooldown_sec", 60)

    for attempt in range(max_retries):
        print(f"\n--- Running benchmark (Attempt {attempt + 1}/{max_retries}) for config: "
              f"num-prompts={config['num_prompts']}, max_curr={config['max_curr']}, input_len={config['input_len']}, output_len={config['output_len']} ---")
        
        command = [
            "vllm", "bench", "serve",
            "--base-url", f"http://{exp_setup['ip']}:{exp_setup['port']}",
            "--backend", "vllm",
            "--model", config["model"],
            "--endpoint", "/v1/completions",
            "--tokenizer", config["tokenizer"],
            "--dataset-name", "random",
            "--random-input-len", str(config["input_len"]),
            "--random-output-len", str(config["output_len"]),
            "--num-prompts", str(config["num_prompts"]),            
            "--percentile-metrics", "ttft,tpot,itl,e2el",
            "--save-result"
        ]
        # Conditionally add arguments that can be None or "inf"
        command.extend(["--request-rate", str(config["req_rate"])])
        if config.get("max_curr") is not None:
            command.extend(["--max-concurrency", str(config["max_curr"])])
        if config.get("goodput"):
            command.append("--goodput")
            command.extend(config["goodput"].split())

        print(f"Executing: {' '.join(command)}")

        try:
            files_before = set(glob.glob("vllm-*.json"))
            start_time = time.time()
            # By removing `capture_output=True`, the subprocess output is streamed
            # to the console, showing real-time progress from the vllm command.
            subprocess.run(command, check=True, text=True)
            end_time = time.time()
            files_after = set(glob.glob("vllm-*.json"))

            new_files = files_after - files_before
            if not new_files:
                print("Error: Benchmark ran, but no new result file (vllm-*.json) was found.")
                continue  # Go to next retry attempt

            result_file = new_files.pop()
            print(f"--- Benchmark run took {end_time - start_time:.2f} seconds. ---")
            print(f"--- Found result file: {result_file} ---")
            
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            # --- Success Condition Check ---
            completed = results.get("completed", 0)
            num_prompts = config.get("num_prompts", 1) # Avoid division by zero
            failed_requests = num_prompts - completed
            
            if failed_requests < num_prompts // 200: # Less than 0.5% failure rate
                print(f"--- Benchmark successful: {completed}/{num_prompts} requests completed (failure rate: {failed_requests/num_prompts:.2%}). ---")
                # Move successful result to archive
                os.makedirs(raw_results_dir, exist_ok=True)
                archive_filepath = os.path.join(raw_results_dir, os.path.basename(result_file))
                os.rename(result_file, archive_filepath)
                print(f"--- Raw results saved to: {archive_filepath} ---")
                return results
            else:
                print(f"--- Benchmark failed: {completed}/{num_prompts} requests completed (failure rate: {failed_requests/num_prompts:.2%}, exceeding threshold). ---")
                os.remove(result_file) # Clean up partial result file

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error running benchmark subprocess: {e}")
            # Stderr is now streamed directly to the console, so any error messages
            # from the child process will be visible above this message.
        # If the attempt failed, wait before retrying
        if attempt < max_retries - 1:
            print(f"Cooldown for {gpu_cooldown_sec} seconds before retry...")
            time.sleep(gpu_cooldown_sec)

    # If all retries fail
    print(f"--- Benchmark failed after {max_retries} attempts. Logging to {failed_runs_file} ---")
    log_failed_run(config, failed_runs_file)
    return None

def main(args):
    """
    Main function to load config, loop through benchmarks, and save results.
    """
    # --- Load Configuration from YAML ---
    try:
        with open(args.config_file, 'r') as f:
            full_config = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading or parsing YAML file {args.config_file}: {e}")
        sys.exit(1)

    exp_setup = full_config.get("experiment_setup", {})
    base_config = full_config.get("base_config", {})
    sweep_params = full_config.get("parameter_sweep", {})

    if not all([exp_setup, base_config, sweep_params]):
        print("Error: YAML file is missing one or more required top-level keys: 'experiment_setup', 'base_config', 'parameter_sweep'")
        sys.exit(1)

    # --- Setup Experiment Directories and Files ---
    short_experiment_name = exp_setup.get("short_experiment_name", f"exp_{datetime.now().strftime('%Y%m%d')}")
    experiment_dir = os.path.join("experiments", short_experiment_name)
    results_csv_file = os.path.join(experiment_dir, "benchmark_results_v2.csv")
    failed_runs_file = os.path.join(experiment_dir, "failed_runs.json")
    raw_results_dir = os.path.join(experiment_dir, "raw_results")

    os.makedirs(experiment_dir, exist_ok=True)

    # --- Generate Benchmark Configurations ---
    benchmark_configs = generate_benchmark_configs(base_config, sweep_params)
    if not benchmark_configs:
        print("No benchmark configurations were generated. Check your parameter_sweep values in the YAML.")
        sys.exit(0)
    
    print(f"Generated {len(benchmark_configs)} benchmark configurations for experiment '{short_experiment_name}'.")

    # Check if the CSV file needs a header.
    write_header = not os.path.exists(results_csv_file) or os.path.getsize(results_csv_file) == 0
    
    with open(results_csv_file, 'a', newline='') as csvfile:
        writer = None

        for i, config in enumerate(benchmark_configs):
            results = run_benchmark(config, exp_setup, raw_results_dir, failed_runs_file)
            
            if not results:
                print(f"--- Skipping results for run {i+1}/{len(benchmark_configs)} due to error ---")
                continue

            # Combine the config and the benchmark results
            combined_data = config.copy()
            combined_data.update(results)
            
            # Add a timestamp for uniqueness
            combined_data["timestamp"] = datetime.now().isoformat()

            # On the first successful run, setup the CSV writer and write the header
            if writer is None:
                # Define column order for better readability in the CSV.
                # Start with base config keys in a sensible order.
                base_keys = [
                    "model", "tokenizer", "hardware", "notes", "pd_enabled",
                    "prefill_node", "prefill_dp", "prefill_tp",
                    "decode_node", "decode_dp", "decode_tp"
                ]
                # Add sweep parameters and other config keys that are not in base_keys.
                other_config_keys = [k for k in config.keys() if k not in base_keys]
                result_keys = list(results.keys())

                fieldnames = base_keys + other_config_keys + result_keys + ["timestamp"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                    write_header = False # Prevent writing header again in this session
            
            # Ensure all fields are present for this row
            row_to_write = {field: combined_data.get(field) for field in writer.fieldnames}
            writer.writerow(row_to_write)
            csvfile.flush() # Save progress immediately
            print(f"--- Successfully saved results for run {i+1}/{len(benchmark_configs)} ---")
            
            # Cooldown between runs, but not after the last one
            if i < len(benchmark_configs) - 1:
                gpu_cooldown_sec = exp_setup.get("gpu_cooldown_sec", 60)
                print(f"GPU cooldown for {gpu_cooldown_sec} seconds...")
                time.sleep(gpu_cooldown_sec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vLLM benchmarks from a YAML configuration file.")
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the YAML configuration file for the experiment."
    )
    args = parser.parse_args()
    main(args)
