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
        "description": """Identify a few of the most relevant categories related to the user's query.\n
                          Only choose categories from the list provided.\n
                          You may choose the same category more than once."""
                          ,
        "parameters": {
            "type": "object",
            "properties": {
                "category_one": {
                    "type": "string",
                    "description": "The most relevant category given the user's query.",
                },
                "category_two": {
                    "type": "string",
                    "description": "The second most relevant category given the user's query.",
                },
                "category_three": {
                    "type": "string",
                    "description": "The third most relevant category given the user's query.",
                },
            },
            "required": ["category_one","category_two","category_three"]
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
        "description": "Tool to add items to the user's cart. These items must be proper nouns from the provided context.",
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {
                    "type": "string",
                    "description": "The COMPLETE product name exactly as it appears in our catalog. If user says partial names like 'LEAFBROOK table', you must extract the full product name 'LEAFBROOK coffee table'. Always include descriptive words like 'coffee table', 'floor lamp', 'dining chair', etc. Examples: 'LEAFBROOK coffee table', 'FOTO floor lamp', 'BIRCHWOOD dining chair'.",
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