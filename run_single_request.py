import requests
import datetime
import random
import argparse
import json

# example command
# python run_single_request.py --endpoint ${ENDPOINT}/v1/completions --model Qwen/Qwen3-235B-A22B --num-words 30000 --max-tokens 2000 &
# 

# Simple concurrency
# CMD_TO_RUN="python run_single_request.py --endpoint ${ENDPOINT}/v1/completions --model Qwen/Qwen3-235B-A22B --num-words 5000 --max-tokens 250"
# PARALLEL_JOBS=512
# seq 1 512 | xargs -P ${PARALLEL_JOBS} -n 1 sh -c "${CMD_TO_RUN}"

SAMPLE_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I", "it", "for",
    "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his",
    "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my",
    "one", "all", "would", "there", "their", "what", "so", "up", "out", "if",
    "about", "who", "get", "which", "go", "me", "when", "make", "can", "like",
    "time", "no", "just", "him", "know", "take", "person", "into", "year",
    "your", "good", "some", "could", "them", "see", "other", "than", "then",
    "now", "look", "only", "come", "its", "over", "think", "also", "back",
    "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"
]

def generate_prompt(num_words):
    """Generates a prompt string with a specified number of random words."""
    print(f"Generating a prompt with {num_words} random words...")
    prompt_words = random.choices(SAMPLE_WORDS, k=num_words)
    return " ".join(prompt_words)

def send_request(endpoint, model, prompt, max_tokens):
    """Sends the completion request and prints timing information."""
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "max_tokens": max_tokens,
        "prompt": prompt,
        "stream": False # Explicitly disable streaming for a single response
    }

    print(f"==================================================")
    print(f"Preparing request at: {datetime.datetime.now().isoformat()}")
    print(f"Target Endpoint: {endpoint}")
    print(f"Model: {model}, Prompt Words: {len(prompt.split())}, Max Tokens: {max_tokens}")
    print(f"==================================================")

    try:
        # The time measurement for the request itself starts inside `requests.post`
        response = requests.post(endpoint, headers=headers, data=json.dumps(data), timeout=300) # 5 min timeout
        
        # Record end time as soon as the call returns
        end_time = datetime.datetime.now()
        
        # `response.elapsed` is a timedelta of time between sending the request
        # and the arrival of the response's headers. This is more accurate.
        duration_td = response.elapsed
        start_time = end_time - duration_td

        print(f"\n==================================================")
        print(f"Request Start: {start_time.isoformat()}")
        print(f"Request End:   {end_time.isoformat()}")
        print(f"Total Duration (from requests lib): {duration_td.total_seconds():.6f} seconds")
        print(f"==================================================")

        if response.status_code == 200:
            print("\nResponse JSON (truncated):")
            try:
                response_data = response.json()
                # Truncate the choice text for clean logging
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    response_data["choices"][0]["text"] = response_data["choices"][0]["text"][:80] + "..."
                print(json.dumps(response_data, indent=2))
            except json.JSONDecodeError:
                print("Could not decode JSON response. Raw text:")
                print(response.text[:200] + "...")
        else:
            print(f"\nError: Received Status Code {response.status_code}")
            print("Response Text:")
            print(response.text)

    except requests.exceptions.ConnectionError as e:
        print(f"\nRequest Failed: Connection Error. Is the server running at {endpoint}?")
        print(f"Error details: {e}")
    except requests.exceptions.ReadTimeout as e:
        print(f"\nRequest Failed: Read Timeout.")
        print(f"Error details: {e}")
    except Exception as e:
        end_time = datetime.datetime.now()
        print(f"\nAn unexpected error occurred at {end_time.isoformat()}: {e}")

def main():
    parser = argparse.ArgumentParser(description="vLLM load testing script.")
    parser.add_argument(
        "--num-words",
        type=int,
        default=12000,
        help="Number of random words to generate for the prompt."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="The max_tokens parameter for the completion request."
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/v1/completions",
        help="The vLLM server endpoint."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-235B-A22B",
        help="The model name to send in the request."
    )
    
    args = parser.parse_args()

    prompt = generate_prompt(args.num_words)
    send_request(args.endpoint, args.model, prompt, args.max_tokens)

if __name__ == "__main__":
    main()