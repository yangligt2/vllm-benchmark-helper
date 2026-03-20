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

def generate_benchmark_configs(vllm_args: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generates a list of benchmark configurations doing a 1D sweep over arrays."""
    list_keys = []
    list_length = 0
    
    for k, v in vllm_args.items():
        if isinstance(v, list):
            list_keys.append(k)
            if list_length == 0:
                list_length = len(v)
            elif len(v) != list_length:
                print(f"Error: All list arguments in vllm_args must have the same length for 1D sweeping. "
                      f"'{k}' has length {len(v)}, expected {list_length}.")
                sys.exit(1)
                
    configs = []
    if list_length == 0:
        # No sweep, just one config
        configs.append(vllm_args.copy())
    else:
        # 1D sweep
        for i in range(list_length):
            config = vllm_args.copy()
            for k in list_keys:
                config[k] = vllm_args[k][i]
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


def run_benchmark(config: Dict[str, Any], metadata: Dict[str, Any], raw_results_dir: str, failed_runs_file: str, log_file: str = None, sweep_info: str = "") -> Dict[str, Any]:
    """
    Runs a single benchmark using the provided config and returns the results.
    Retries on failure (completed != num-prompts).
    """
    max_retries = metadata.get("max_retries", 3)
    gpu_cooldown_sec = metadata.get("gpu_cooldown_sec", 60)

    for attempt in range(max_retries):
        print(f"\n--- Running benchmark (Attempt {attempt + 1}/{max_retries}) ---")
        
        command = ["vllm", "bench", "serve"]
        
        # Use ip and port from metadata if base-url is not explicitly provided in vllm_args
        if "base-url" not in config:
            ip = metadata.get("ip", "localhost")
            port = metadata.get("port", 8000)
            command.extend(["--base-url", f"http://{ip}:{port}"])
        
        for k, v in config.items():
            if isinstance(v, bool):
                if v:
                    command.append(f"--{k}")
            else:
                if k == "header":
                    command.extend(["--header", str(v)])
                elif k == "goodput":
                    # goodput takes multiple arguments separated by spaces conceptually, e.g. ttft:1000 tpot:40
                    command.append("--goodput")
                    command.extend(str(v).split())
                else:
                    command.extend([f"--{k}", str(v)])

        print(f"Executing: {' '.join(command)}")

        try:
            backend = config.get("backend", "vllm")
            file_pattern = f"{backend}-*.json"
            files_before = set(glob.glob(file_pattern))
            start_time = time.time()
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            output_lines = []
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                output_lines.append(line)
            process.wait()
            end_time = time.time()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command, "".join(output_lines))
                
            # Extract the benchmark result block
            bench_body = []
            capture = False
            for line in output_lines:
                if "============ Serving Benchmark Result ============" in line:
                    capture = True
                if capture:
                    bench_body.append(line)
                if capture and "==================================================" in line:
                    break
            
            if bench_body and log_file:
                with open(log_file, "a") as lf:
                    lf.write(f"############################\n")
                    lf.write(f"#### {sweep_info}\n")
                    lf.write(f"############################\n")
                    lf.write("".join(bench_body))
                    lf.write("\n")
            
            files_after = set(glob.glob(file_pattern))

            new_files = files_after - files_before
            if not new_files:
                print(f"Error: Benchmark ran, but no new result file ({file_pattern}) was found.")
                continue  # Go to next retry attempt

            result_file = new_files.pop()
            print(f"--- Benchmark run took {end_time - start_time:.2f} seconds. ---")
            print(f"--- Found result file: {result_file} ---")
            
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            # --- Success Condition Check ---
            completed = results.get("completed", 0)
            
            # Find the number of prompts either via num-prompts directly or default to 1 for calculation
            num_prompts = config.get("num-prompts", config.get("num_prompts", 1))
            try:
                num_prompts = int(num_prompts)
            except ValueError:
                num_prompts = 1
                
            failed_requests = num_prompts - completed
            
            if failed_requests < num_prompts // 5: # Less than 20% failure rate
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

    vllm_args = full_config.get("vllm_args", {})
    if not vllm_args:
        print("Error: YAML file is missing 'vllm_args'.")
        sys.exit(1)

    # Everything outside `vllm_args` goes directly into the CSV as metadata 
    # (except for nested structures that aren't useful, but we flatten typically)
    metadata = {k: v for k, v in full_config.items() if k != "vllm_args"}

    # --- Setup Experiment Directories and Files ---
    short_experiment_name = metadata.get("short_experiment_name", f"exp_{datetime.now().strftime('%Y%m%d')}")
    experiment_dir = os.path.join("experiments", short_experiment_name)
    results_csv_file = os.path.join(experiment_dir, "benchmark_results_v2.csv")
    failed_runs_file = os.path.join(experiment_dir, "failed_runs.json")
    raw_results_dir = os.path.join(experiment_dir, "raw_results")
    
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(experiment_dir, f"benchmark_{timestamp_str}.log")

    os.makedirs(experiment_dir, exist_ok=True)

    # Find keys that were swept
    list_keys = [k for k, v in vllm_args.items() if isinstance(v, list)]

    # --- Generate Benchmark Configurations ---
    benchmark_configs = generate_benchmark_configs(vllm_args)
    if not benchmark_configs:
        print("No benchmark configurations were generated.")
        sys.exit(0)
    
    print(f"Generated {len(benchmark_configs)} benchmark configurations for experiment '{short_experiment_name}'.")

    # Check if the CSV file needs a header.
    write_header = not os.path.exists(results_csv_file) or os.path.getsize(results_csv_file) == 0
    
    with open(results_csv_file, 'a', newline='') as csvfile:
        writer = None

        for i, config in enumerate(benchmark_configs):
            sweep_info_parts = [f"{k} {config.get(k, '')}" for k in list_keys]
            sweep_info = ", ".join(sweep_info_parts) if sweep_info_parts else "base run"
            
            results = run_benchmark(config, metadata, raw_results_dir, failed_runs_file, log_file, sweep_info)
            
            if not results:
                print(f"--- Skipping results for run {i+1}/{len(benchmark_configs)} due to error ---")
                continue

            # Combine the metadata, config and the benchmark results
            combined_data = metadata.copy()
            combined_data.update(config)
            combined_data.update(results)
            
            # Add a timestamp for uniqueness
            combined_data["timestamp"] = datetime.now().isoformat()

            # On the first successful run, setup the CSV writer and write the header
            if writer is None:
                fieldnames = list(combined_data.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                    write_header = False # Prevent writing header again in this session
            
            # Ensure all fields are present for this row (if fieldnames were added later, it skips them)
            # Add missing keys to fieldnames dynamically? No, DictWriter is fixed.
            # But combined_data maps well.
            row_to_write = {}
            for field in writer.fieldnames:
                row_to_write[field] = combined_data.get(field)
                
            writer.writerow(row_to_write)
            csvfile.flush() # Save progress immediately
            print(f"--- Successfully saved results for run {i+1}/{len(benchmark_configs)} ---")
            
            # Cooldown between runs, but not after the last one
            if i < len(benchmark_configs) - 1:
                gpu_cooldown_sec = metadata.get("gpu_cooldown_sec", 60)
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
