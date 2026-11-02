# Embedding Management for Catalog Retriever

## 📋 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Force Repopulation](#force-repopulation)
- [When to Repopulate](#when-to-repopulate)
- [Custom Data Source](#custom-data-source)

## Overview

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

# Preserve the bind-mounted data, then let Compose create a fresh directory
mv catalog_retriever/volumes catalog_retriever/volumes.backup

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

## Custom Data Source

The application comes with sample product data (`products_extended.csv`), but you can easily replace it with your own product catalog. This section explains how to use a custom CSV file for your retail data.

### CSV File Format

Your custom CSV file should include the following columns:

| Column | Description | Required | Example |
|--------|-------------|----------|---------|
| `name` | Product name | Yes | "Classic Black Patent Leather Purse" |
| `description` | Product description | Yes | "Elegant black patent leather purse..." |
| `category` | Product category | Yes | "bag" |
| `subcategory` | Product subcategory | Yes | "handbag" |
| `url` | Product page or display URL | No | "/images/purse_image.jpg" |
| `price` | Numeric product price | Yes | "89.99" |
| `image` | Image URL or path under `shared/` | Yes | "/images/purse_image.jpg" |

Additional columns are preserved as product metadata, but the required column
names above must match exactly.

### Step-by-Step Guide

#### Step 1: Prepare Your Data File

1. **Create your CSV file** with your product data:
   ```bash
   # Example: my_products.csv
   category,subcategory,name,description,url,price,image
   "shoes","sneakers","Custom Product 1","Description of product 1","/images/product1.jpg",99.99,"/images/product1.jpg"
   "bags","handbags","Custom Product 2","Description of product 2","/images/product2.jpg",149.99,"/images/product2.jpg"
   ```

2. **Add the CSV file** to the shared data directory:
   ```bash
   # Copy your CSV file to the shared data directory
   cp my_products.csv shared/data/
   ```

#### Step 2: Update Configuration

1. **Edit the catalog retriever configuration**:
   ```bash
   # Edit the configuration file
   shared/configs/catalog_retriever/config.yaml
   ```

2. **Update the data_source parameter**:
   ```yaml
   data_source: "/app/shared/data/my_products.csv"  # Update this line
   ```

#### Step 3: Clear Vector Database Cache

1. **Preserve the existing vector database and start with empty state**:
   ```bash
   # Stop the services first
   docker compose -f docker-compose.yaml down
   
   # Move the bind-mounted state so it can be recovered if needed
   mv catalog_retriever/volumes catalog_retriever/volumes.previous
   ```

#### Step 4: Restart Services

**Restart the application** to use the new data:
```bash
# Restart services to rebuild the vector database
docker compose -f docker-compose.yaml up -d --build

# Monitor the catalog retriever logs to see indexing progress
docker compose logs -f catalog-retriever
```

### Adding Product Images

If your products have images:

1. **Add image files** to the shared images directory:
   ```bash
   # Copy your product images
   shared/images/
   ```

2. **Update image URLs** in your CSV to reference the filenames:
   ```csv
   category,subcategory,name,description,url,price,image
   "shoes","sneakers","Product 1","Description","/images/product1.jpg",99.99,"/images/product1.jpg"
   ```

3. **Clear the existing collections or move the bind-mounted database state**, as
   described under [Force Repopulation](#force-repopulation), then start the
   application to rebuild both text and image embeddings:
   ```bash
   docker compose -f docker-compose.yaml up -d --build
   ```

### Configuration for Different Environments

For different environments, you can create override configs:

```bash
# Create custom override
cp shared/configs/catalog_retriever/config.yaml shared/configs/catalog_retriever/config-custom.yaml

# Edit the custom config with your custom data source
```

Then use it:
```bash
export CONFIG_OVERRIDE=config-custom.yaml
docker compose -f docker-compose.yaml up -d --build
```

### Verification

After restarting, verify your custom data is loaded:

```bash
# Check catalog retriever health inside its private container network
docker compose -f docker-compose.yaml exec catalog-retriever \
  python -c 'import urllib.request; print(urllib.request.urlopen("http://localhost:8010/health").read().decode())'

# Check if your products are searchable
docker compose -f docker-compose.yaml exec catalog-retriever \
  python -c 'import json, urllib.request; data=json.dumps({"text":["your_product_name"],"k":5}).encode(); req=urllib.request.Request("http://localhost:8010/query/text",data=data,headers={"Content-Type":"application/json"}); print(urllib.request.urlopen(req).read().decode())'
```
