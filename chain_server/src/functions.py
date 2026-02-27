# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

retrieval_extraction_function = {
    "type": "function",
    "function": {
        "name": "extract_retrieval_inputs",
        "description": """Extract structured retrieval inputs from the user request.
                          Return:
                          - search_entities for retrieval queries
                          - up to three relevant categories from the provided category list
                          - explicit numeric price filters when provided by the user
                          Do not infer missing constraints.

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
                    "items": {"type": "string"}
                },
                "category_one": {
                    "type": "string",
                    "description": "Most relevant category from available categories."
                },
                "category_two": {
                    "type": "string",
                    "description": "Second most relevant category from available categories."
                },
                "category_three": {
                    "type": "string",
                    "description": "Third most relevant category from available categories."
                },
                "min_price": {
                    "type": "number",
                    "description": "Minimum price requested by the user, if explicitly provided."
                },
                "max_price": {
                    "type": "number",
                    "description": "Maximum price requested by the user, if explicitly provided."
                }
            },
            "required": ["search_entities", "category_one", "category_two", "category_three"]
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
                    "description": "The name of the item. Must be from the chat history, or most recent user query.",
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
