# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
A function that interprets relevant categories based on the user's query.
"""
search_function = {
    "type": "function",
    "function": {
        "name": "search_entities",
        "description": """Extract search terms for product catalog search.
                          
                          IMPORTANT: 
                          - For NEW product searches, extract only the new product type being requested
                          - For questions about PREVIOUSLY mentioned products, extract the specific product name from context
                          - NEVER combine or merge context products with new search terms""",
        "parameters": {
            "type": "object",
            "properties": {
                "search_entities": {
                    "type": "array",
                    "description": "Individual terms that the user is searching for.",
                    "items":{
                        "type": "string"
                    }
                }
            },
            "required": ["search_entities"]
        }
    }
}

category_function = {
    "type": "function",
    "function": {
        "name": "get_categories",
        "description": """Identify exactly 3 relevant categories for the user's query from the provided list."""
                          ,
        "parameters": {
            "type": "object",
            "properties": {
                "categories": {
                    "type": "array",
                    "description": "Most relevant categories from the provided list.",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["categories"]
        }
    }
}

"""
A function that responds to the user and summarizes the context.
"""
summary_function = {
    "type" : "function",
    "function" : {
        "name" : "summarizer",
        "description" : "Tool that summarizes the context of the user's conversation.",
        "parameters" : {
            "type" : "object",
            "properties" : {
                "summary" : {
                    "type" : "string",
                    "description" : "A concise summary that MUST preserve: all product names, product specifications (materials, colors, care instructions, prices), products the user asked about, and cart contents. Summarize only the general conversation flow and user preferences."
                },
            },
            "required" : ["summary"]
        },
    },
}

"""
Gets items to add to the users cart.
"""
add_to_cart_function = {
    "type": "function",
    "function": {
        "name": "add_to_cart",
        "description": "Tool to add items to the user's cart. When the user refers to 'it', 'this', 'that', or other pronouns, extract the specific product name from the provided context.",
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {
                    "type": "string",
                    "description": "The exact name of the item to add. If the user query contains pronouns like 'it', 'this', 'that', extract the specific product name from the context. Example: if context mentions 'Ginger Lace Trim Gown' and user says 'add it to cart', use 'Ginger Lace Trim Gown' as the item_name.",
                },
                "quantity": {
                    "type": "integer",
                    "description": "The number of items to add to the cart.",
                },
            },
            "required": ["item_name", "quantity"],
        },
    },
}

"""
Removes items from the user's cart.
"""
remove_from_cart_function = {
    "type": "function",
    "function": {
        "name": "remove_from_cart",
        "description": "Tool to remove items to the user's cart.",
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {
                    "type": "string",
                    "description": "The name of item to add to the cart.",
                },
                "quantity": {
                    "type": "integer",
                    "description": "The number of items to add to the cart.",
                },
            },
            "required": ["item_name", "quantity"],
        },
    },
}

"""
Views items in the user's cart.
"""
view_cart_function = {
    "type": "function",
    "function": {
        "name": "view_cart",
        "description": "Tool to view the user's cart.",
    },
}