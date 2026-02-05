# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import time

# Use /app/data for writable storage in containerized environments
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:////app/data/context.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    context = Column(String, default="")

class CartItem(Base):
    __tablename__ = "cart_items"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    item = Column(String)
    amount = Column(Integer)

Base.metadata.create_all(bind=engine)

class ContextUpdate(BaseModel):
    new_context: str

class ItemUpdate(BaseModel):
    item: str
    amount: int

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/user/{user_id}")
async def get_user(user_id: int):
    db = SessionLocal()
    user = db.query(User).filter(User.id == user_id).first()
    cart_items = db.query(CartItem).filter(CartItem.id == user_id).all()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": user.id, "context": user.context, "cart": [{"item": item.item, "amount": item.amount} for item in cart_items]}

@app.get("/user/{user_id}/cart")
async def report_cart(user_id: int):
    db = SessionLocal()
    cart_items = db.query(CartItem).filter(CartItem.user_id == user_id).all()
    if not cart_items:
        return {
            "user_id": user_id,
            "cart": []
        }      
    else:
        return {
            "user_id": user_id,
            "cart": [{"item": item.item, "amount": item.amount} for item in cart_items]
        }
  
@app.get("/user/{user_id}/context")
async def get_context(user_id: int):
    db = SessionLocal()
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return {
            "user_id": user_id,
            "context" : ""
        }
    else:
        return {
            "user_id": user_id,
            "context" : user.context
        }

@app.post("/user/{user_id}/cart/add")
async def add_to_cart(user_id: int, item_update: ItemUpdate):
    db = SessionLocal()
    item = item_update.item
    amount = item_update.amount
    cart_item = db.query(CartItem).filter(CartItem.user_id == user_id, CartItem.item == item).first()
    if cart_item:
        cart_item.amount += amount
    else:
        cart_item = CartItem(user_id=user_id, item=item, amount=amount)
        db.add(cart_item)
    db.commit()
    return {
        "user_id": user_id,
        "message": f"In response to the user's request, I have added {amount} of '{item}' to their cart."
        }

@app.post("/user/{user_id}/cart/remove")
async def remove_cart(user_id: int, item_update: ItemUpdate):
    db = SessionLocal()
    item = item_update.item
    amount = item_update.amount
    cart_item = db.query(CartItem).filter(CartItem.user_id == user_id, CartItem.item == item).first()
    if not cart_item:
        raise HTTPException(status_code=404, detail="Item not in cart")
    if cart_item.amount <= amount:
        db.delete(cart_item)
    else:
        cart_item.amount -= amount
    db.commit()
    return {
        "user_id": user_id,
        "message": f"In response to the user's request, I have removed {amount} of '{item}' from cart."
        }

@app.post("/user/{user_id}/cart/clear")
async def clear_cart(user_id: int):
    db = SessionLocal()
    cart_items = db.query(CartItem).filter(CartItem.user_id == user_id).all()
    if not cart_items:
        raise HTTPException(status_code=404, detail="No items found in cart")
    for item in cart_items:
        db.delete(item)
    db.commit()
    return {
        "user_id": user_id,
        "message": f"In response to the user's request, the cart for user {user_id} has been deleted."
        }

@app.post("/user/{user_id}/context/add")
async def add_context(user_id: int, context_update: ContextUpdate):
    db = SessionLocal()
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        user = User(id=user_id, context=context_update.new_context)
        db.add(user)
    else:
        user.context += " " + context_update.new_context
    db.commit()
    return {
        "user_id": user_id,
        "message": "Context updated successfully"
        }

@app.post("/user/{user_id}/context/replace")
async def replace_context(user_id: int, context_update: ContextUpdate):
    db = SessionLocal()
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        user = User(id=user_id, context=context_update.new_context)
        db.add(user)
    else:
        user.context = context_update.new_context
    db.commit()
    return {
        "user_id": user_id,
        "message": "Context updated successfully"
        }

@app.post("/user/{user_id}/context/clear")
async def clear_context(user_id: int):
    db = SessionLocal()
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    return {
        "user_id": user_id,
        "message": f"In response to the user's request, context for user {user_id} has been deleted."
        }

@app.post("/user/{user_id}/clear")
async def clear_user(user_id: int):
    db = SessionLocal()
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    return {
        "user_id": user_id,
        "message": f"In response to the user's request, deleted cart and context for user {user_id}"
        }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }