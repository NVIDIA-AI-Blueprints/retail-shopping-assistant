import argparse
import yaml
import requests
import time
import os
import random
from uuid import uuid4

parser = argparse.ArgumentParser(
                    prog='ConvTest',
                    description='Tests the conversation API')

parser.add_argument('-p', '--port', default=3000, type=int)
parser.add_argument('-H', '--host', default='localhost', type=str)
parser.add_argument('-d', '--result_directory', default='results')
parser.add_argument('-u', '--uri', default='api/query/timing')
parser.add_argument(
    '--disable-guardrails',
    action='store_true',
    help='Send guardrails=false in every query payload.',
)
parser.add_argument(
    '--request-timeout',
    default=120,
    type=float,
    help='Seconds to wait for each API request before recording an error.',
)

args = parser.parse_args()

# CONFIGURATION
sub_path = os.environ["TEST_PATH"]
INPUT_DIRECTORY = f"conversations/{sub_path}"
OUTPUT_DIRECTORY = f"{INPUT_DIRECTORY}/{args.result_directory}"
API_ENDPOINT = f"http://{args.host}:{args.port}/{args.uri}"
REQUEST_DELAY = 0.5

# Ensure the output directory exists
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# Collect all YAML files in the directory
yaml_files = [f for f in os.listdir(INPUT_DIRECTORY) if f.endswith('.yaml') or f.endswith('.yml')]

for filename in yaml_files:

    user_id = random.randint(0,99999)
    scope_id = uuid4().hex
    session_id = f"integration-session-{scope_id}"
    conversation_id = f"integration-conversation-{scope_id}"
    cart_id = f"integration-cart-{scope_id}"

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

    for query_index, query in enumerate(queries):
        payload = {
            "user_id" : user_id,
            "query": query,
            "guardrails": not args.disable_guardrails,
            "session_id": session_id,
            "conversation_id": conversation_id,
            "cart_id": cart_id,
            "request_id": f"integration-request-{scope_id}-{query_index}",
            }
        try:
            response = requests.post(API_ENDPOINT, json=payload, timeout=args.request_timeout)
            response.raise_for_status()
            data = response.json()
            response_text = data.get("response", "No response collected.")
            results.append({
                "query": query,
                "content": response_text,
                "response": response_text,
                "timing": data.get("timings", "No timing collected." ),
                "model_usage": data.get("model_usage", {}),
                "agent_diagnostics": data.get("agent_diagnostics", {}),
            })
        except Exception as e:
            results.append({
                "query": query,
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
