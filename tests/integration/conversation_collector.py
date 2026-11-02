import argparse
import math
import yaml
import requests
import time
import os
import random


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a nonnegative number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite, nonnegative number")
    return parsed


parser = argparse.ArgumentParser(
                    prog='ConvTest',
                    description='Tests the conversation API')

parser.add_argument('-p', '--port', default=3000, type=int)
parser.add_argument('-H', '--host', default='localhost', type=str)
parser.add_argument('-d', '--result_directory', default='results')
parser.add_argument('-u', '--uri', default='api/query/timing')
parser.add_argument(
    '--request-delay',
    default=0.5,
    type=_nonnegative_float,
    help='Seconds to wait between requests. Defaults to 0.5.',
)

args = parser.parse_args()

# CONFIGURATION
sub_path = os.environ["TEST_PATH"]
INPUT_DIRECTORY = f"conversations/{sub_path}"
OUTPUT_DIRECTORY = f"{INPUT_DIRECTORY}/{args.result_directory}"
API_ENDPOINT = f"http://{args.host}:{args.port}/{args.uri}"
REQUEST_TIMEOUT = 210

# Ensure the output directory exists
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# Collect all YAML files in the directory
yaml_files = sorted(
    f
    for f in os.listdir(INPUT_DIRECTORY)
    if f.endswith('.yaml') or f.endswith('.yml')
)
request_errors = 0

for filename in yaml_files:

    user_id = random.randint(0,99999)

    print(f"USER_ID: {user_id}")
    
    input_path = os.path.join(INPUT_DIRECTORY, filename)
    output_filename = filename.replace('.yaml', '.yaml').replace('.yml', '.yml')
    output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)

    with open(input_path, 'r') as f:
        query_set = yaml.safe_load(f)

    print(query_set)

    set_name = query_set.get('set_name', filename)
    queries = query_set.get('queries', [])
    results = []

    print(f"Processing file: {filename} (set: {set_name})")

    for query in queries:
        payload = {
            "user_id" : user_id,
            "query": query
            }
        try:
            response = requests.post(
                API_ENDPOINT,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            results.append({
                "query": query,
                "content": data.get("content", "No response collected."),
                "response": data.get("response", "No response collected."),
                "timing": data.get("timings", "No timing collected." )
            })
        except Exception as e:
            request_errors += 1
            results.append({
                "query": query,
                "response": f"Error: {str(e)}"
            })
        time.sleep(args.request_delay)

    # Save individual result
    with open(output_path, 'w') as f:
        yaml.dump({
            "set_name": set_name,
            "results": results
        }, f, sort_keys=False)

    print(f"Saved results to {output_path}")

if request_errors:
    raise SystemExit(
        f"Integration collection failed with {request_errors} request error(s)."
    )
