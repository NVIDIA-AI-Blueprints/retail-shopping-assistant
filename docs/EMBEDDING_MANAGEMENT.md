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

## Custom Data Source

The application comes with sample product data (`products_extended.csv`), but you can easily replace it with your own product catalog. This section explains how to use a custom CSV file for your retail data.

### CSV File Format

Your custom CSV file should include product text fields and any metadata fields
you want the catalog retriever to expose as hard filters. The default
fashion catalog uses the columns below, but future catalogs can use different
filter columns when `shared/configs/catalog_retriever/config.yaml` declares
them in `filter_registry`.

Do not copy enum values such as colors, materials, sizes, or categories into
configuration. Declare the CSV field names in `filter_registry`; the catalog
retriever discovers the actual values after it loads the CSV. See
[Catalog Filter Configuration](CATALOG_FILTERS.md) for the detailed workflow.

| Column | Description | Required | Example |
|--------|-------------|----------|---------|
| `name` | Product name | Yes | "Classic Black Patent Leather Purse" |
| `description` | Product description | Yes | "Elegant black patent leather purse..." |
| `category` / `subcategory` | Default catalog category metadata | No | "bag" |
| `brand` | Product brand | No | "Fashion Brand" |
| `price` | Product price | No | "89.99" |
| custom metadata | Fields declared as filters | No | "green", "cotton" |
| `image` | Product image URL or filename | No | "purse_image.jpg" |

### Step-by-Step Guide

#### Step 1: Prepare Your Data File

1. **Create your CSV file** with your product data:
   ```bash
   # Example: my_products.csv
   name,description,category,subcategory,color,brand,price,image
   "Custom Product 1","Description of product 1","fashion","shoes","black","Brand A","99.99","product1.jpg"
   "Custom Product 2","Description of product 2","fashion","bag","green","Brand B","149.99","product2.jpg"
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

1. **Remove the existing vector database volumes**:
   ```bash
   # Stop the services first
   docker compose -f docker-compose.yaml down
   
   # Remove the catalog retriever volumes to force re-indexing
   rm -rf catalog_retriever/volumes/
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
   name,description,subcategory,image
   "Product 1","Description","shoes","product1.jpg"
   ```

3. **Restart services** to index the new images:
   ```bash
   docker compose -f docker-compose.yaml restart catalog-retriever
   ```

### Configuration for Different Environments

For different model endpoints, edit the relevant role in
`shared/configs/models.yaml`:

```bash
python scripts/model_config.py show --validate
```

For a custom product CSV location, set `CATALOG_DATA_SOURCE` and restart the
catalog service:

```bash
export CATALOG_DATA_SOURCE=/app/shared/data/products.csv
docker compose -f docker-compose.yaml up -d --build
```

### Verification

After restarting, verify your custom data is loaded:

```bash
# Check catalog retriever health
curl http://localhost:8010/health

# Check if your products are searchable
curl -X POST http://localhost:8010/search \
  -H "Content-Type: application/json" \
  -d '{"query": "your_product_name", "top_k": 5}'
```
