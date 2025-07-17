# Embedding Management for Catalog Retriever

The catalog retriever includes simple embedding caching to avoid unnecessary reprocessing on startup.

## How It Works

- **On Startup**: The system checks if embeddings already exist in the Milvus database
- **If Embeddings Exist**: Skips population and uses existing embeddings
- **If No Embeddings**: Populates embeddings from the CSV file

## Force Repopulation

To force the system to repopulate embeddings (e.g., when you change the embedding model, update products.csv or images, etc), you need to delete the existing embeddings from the Milvus database.

### Option 1: Delete via Milvus CLI

```bash
# Connect to Milvus container
docker exec -it <milvus-standalone-container> bash

# Use Milvus CLI to delete collections
milvus_cli
use default
drop collection shopping_advisor_text_db
drop collection shopping_advisor_image_db
exit
```

### Option 2: Delete via Python Script

Create a script to delete the collections:

```python
from pymilvus import connections, utility

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Delete collections
if utility.has_collection("shopping_advisor_text_db"):
    utility.drop_collection("shopping_advisor_text_db")
    print("Text collection deleted")

if utility.has_collection("shopping_advisor_image_db"):
    utility.drop_collection("shopping_advisor_image_db")
    print("Image collection deleted")
```

### Option 3: Restart with Fresh Database

If using Docker Compose, you can restart with a fresh Milvus database:

```bash
# Stop the services
docker compose down

# Remove Milvus volume to start fresh
docker volume rm retail-shopping-assistant_milvus_data

# Restart services
docker compose up -d
```

## When to Repopulate

You should force repopulation when:
- You want to use a different embedding model
- Products.csv file is updated
- Product images are modified
- You want to ensure fresh embeddings
- Database corruption is suspected
