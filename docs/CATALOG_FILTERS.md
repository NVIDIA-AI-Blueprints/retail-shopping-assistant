# Catalog Filters

This guide explains how catalog filter metadata works when you replace or
extend the product catalog.

## Core Rule

Do not hardcode enum values in chain-server, UI, prompts, or docs.

The only place you declare catalog filters is:

```text
shared/configs/catalog_retriever/config.yaml
```

That config declares **which CSV fields are valid hard filters**. It does not
declare enum values such as `dress`, `bag`, `blue`, or `cotton`.

Enum values and numeric ranges are discovered by the catalog retriever after it
loads the configured CSV. The discovered filter contract is exposed through:

```text
http://localhost:8010/capabilities
http://localhost:8009/capabilities
```

## What `filter_registry` Means

`filter_registry` is catalog policy. Each entry answers:

- What is the public filter name?
- What type of filter is it?
- Which CSV column or columns provide its values?
- Which operators are supported?

Example:

```yaml
filter_registry:
  category:
    type: "enum"
    source_fields:
      - "subcategory"
    operators:
      - "in"
  price:
    type: "number"
    source_fields:
      - "price"
    operators:
      - "gte"
      - "lte"
```

This means:

- `category` is an enforceable enum filter.
- Its values are read from the CSV `subcategory` column.
- `price` is an enforceable numeric filter.
- Its range is read from the CSV `price` column.

It does **not** mean the enum values are known ahead of time.

## What You Need To Know Before Deployment

You do not need to know enum values before deployment.

You do need to decide which CSV columns are allowed to become hard filters. That
is a catalog policy decision, not an enum-value decision.

For example, before deployment you might know:

```text
This catalog has columns named color, material, size, and price.
Users may strictly filter on those fields.
```

That is enough to write `filter_registry`. You do not need to know whether the
loaded values are `blue`, `navy`, `teal`, `cotton`, `linen`, or any other
specific value.

If the incoming catalog schema is also unknown, inspect the CSV header before
deployment and decide which fields should be filterable. Do not guess from user
language and do not promote every column automatically. Some columns are product
text, IDs, image paths, long descriptions, or internal metadata and should not
be hard filters.

## Example: New Catalog With Different Fields

Suppose a new catalog CSV contains:

```csv
category,subcategory,name,description,image,price,color,material,size
apparel,dress,Blue Silk Dress,Formal silk dress,/images/blue_dress.jpg,120,blue,silk,M
apparel,dress,Green Cotton Dress,Casual cotton dress,/images/green_dress.jpg,80,green,cotton,L
accessories,bag,Black Leather Tote,Work tote,/images/black_tote.jpg,180,black,leather,one size
```

If users should be able to say "only blue", "only silk", or "size M" and have
the catalog strictly enforce that, declare those fields as filters:

```yaml
filter_registry:
  category:
    type: "enum"
    source_fields:
      - "subcategory"
    operators:
      - "in"
  color:
    type: "enum"
    source_fields:
      - "color"
    operators:
      - "in"
  material:
    type: "enum"
    source_fields:
      - "material"
    operators:
      - "in"
  size:
    type: "enum"
    source_fields:
      - "size"
    operators:
      - "in"
  price:
    type: "number"
    source_fields:
      - "price"
    operators:
      - "gte"
      - "lte"
```

After restart/reindex, `/capabilities` will contain values discovered from the
CSV, for example:

```json
{
  "filters": {
    "category": {
      "type": "enum",
      "source_fields": ["subcategory"],
      "values": ["bag", "dress"]
    },
    "color": {
      "type": "enum",
      "source_fields": ["color"],
      "values": ["black", "blue", "green"]
    },
    "material": {
      "type": "enum",
      "source_fields": ["material"],
      "values": ["cotton", "leather", "silk"]
    },
    "size": {
      "type": "enum",
      "source_fields": ["size"],
      "values": ["L", "M", "one size"]
    },
    "price": {
      "type": "number",
      "source_fields": ["price"],
      "min_value": 80,
      "max_value": 180
    }
  }
}
```

The values above come from the ingested CSV. They are not copied into config.

## How User Requests Use Filters

The language layer builds a structured search request from user intent. The
request builder then validates every requested filter against catalog
capabilities.

Example user request:

```text
Only show me blue silk dresses under 100.
```

If `category`, `color`, `material`, and `price` are declared filters, the search
plan can contain:

```json
{
  "query": "dresses",
  "filters": {
    "category": ["dress"],
    "color": ["blue"],
    "material": ["silk"],
    "price": {"max": 100}
  },
  "strictness": "hard"
}
```

If `color` is not declared as a filter, it is not sent as a hard filter. The
catalog tool must not pretend it can enforce a field the catalog did not
declare.

## Current CSV Loader Expectations

The current catalog loader expects the bundled product rows to include these
core columns for indexing and display:

| Column | Purpose |
| --- | --- |
| `name` | Product display name and retrieval text |
| `description` | Product description and retrieval text |
| `category` | Broad catalog grouping |
| `subcategory` | Default public `category` filter source |
| `image` | Product image path or URL |
| `price` | Numeric price metadata when price filtering is enabled |

Additional columns such as `color`, `material`, `size`, `brand`, or
`department` may be used as filters when declared in `filter_registry`.

## Refresh Checklist

1. Put the new CSV under `shared/data/`.
2. Set `data_source` in `shared/configs/catalog_retriever/config.yaml`.
3. Declare only filter field names, types, source fields, and operators in
   `filter_registry`.
4. Do not write enum values in config.
5. Clear/rebuild catalog embeddings if the product data changed.
6. Restart catalog retriever and chain-server.
7. Check `http://localhost:8010/capabilities`.
8. Check `http://localhost:8009/capabilities`.

## Troubleshooting

If a filter is missing from `/capabilities`, check that it is declared in
`filter_registry`.

If an enum filter appears but its `values` list is empty, check that the
configured `source_fields` exactly match CSV column names and that the CSV rows
contain non-empty values.

If a user asks for a strict filter and results are not filtered, check that the
field is declared in `filter_registry` and appears under `/capabilities.filters`.

If a new catalog uses different core columns for product text or images, update
the catalog loader before deploying that catalog shape.
