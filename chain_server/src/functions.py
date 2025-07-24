"""
A function that interprets relevant categories based on the user's query.
"""
search_function = {
    "type": "function",
    "function": {
        "name": "get_categories",
        "description": """Identify relevant search terms for a user given their most recent query, and their chat history.\n
                          Additionally, determine relevant categories for those search terms given the provided category list.""",
        "parameters": {
            "type": "object",
            "properties": {
                "relevant_categories": {
                    "type": "array",
                    "description": "The most relevant categories from the provided list of categories given the user's query.",
                    "items":{
                        "type": "string"
                    }
                }
            },
            "type": "object",
            "properties": {
                "search_entities": {
                    "type": "array",
                    "description": "Terms that the user is searching for.",
                    "items":{
                        "type": "string"
                    }
                }
            },
            "required": ["search_entities","relevant_categories"]
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
                    "description" : "A short, but detailed summary of the provided context, including chat roles and item names."
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