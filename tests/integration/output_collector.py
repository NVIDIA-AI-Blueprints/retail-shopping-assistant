import argparse
import yaml
import requests
import time
import os

parser = argparse.ArgumentParser(
                    prog='ConvTest',
                    description='Tests the conversation API')

parser.add_argument('-p', '--port', default=8003, type=int)
parser.add_argument('-d', '--result_directory', default='gr_out_results')
parser.add_argument('-u', '--uri', default='rail/output/timing')

args = parser.parse_args()

# CONFIGURATION
INPUT_DIRECTORY = "conversations/initial/results"
OUTPUT_DIRECTORY = f"conversations/initial/{args.result_directory}"
API_ENDPOINT = f"http://localhost:{args.port}/{args.uri}"
REQUEST_DELAY = 0.5
USER_ID = 2

# Ensure the output directory exists
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# Collect all YAML files in the directory
yaml_files = [f for f in os.listdir(INPUT_DIRECTORY) if f.endswith('.yaml') or f.endswith('.yml')]

for filename in yaml_files:
    input_path = os.path.join(INPUT_DIRECTORY, filename)
    output_filename = filename.replace('.yaml', '.results.yaml').replace('.yml', '.results.yml')
    output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)

    with open(input_path, 'r') as f:
        query_set = yaml.safe_load(f)

    set_name = query_set.get('set_name', filename)
    queries = query_set.get('results', [])
    results = []

    print(f"Processing file: {filename} (set: {set_name})")

    for query in queries:
        payload = {
            "user_id" : USER_ID,
            "query": query["response"]
            }
        try:
            response = requests.post(API_ENDPOINT, json=payload)
            response.raise_for_status()
            data = response.json()
            results.append({
                "query": query["response"],
                "content": data.get("content", "No response collected."),
                "response": data.get("response", "No response collected."),
                "timing": data.get("timings", "No timing collected." )
            })
        except Exception as e:
            results.append({
                "query": query["response"],
                "response": f"Error: {str(e)}"
            })
        time.sleep(REQUEST_DELAY)

    # Save individual result
    with open(output_path, 'w') as f:
        yaml.dump({
            "set_name": set_name,
            "results": results
        }, f, sort_keys=False)

    print(f"Saved results to {output_path}")
