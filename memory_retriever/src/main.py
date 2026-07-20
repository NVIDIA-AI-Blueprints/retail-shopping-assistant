# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import Column, Float, Integer, String, create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Optional
import json
import logging
import time
from uuid import uuid4

DATABASE_URL = "sqlite:///./context.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


def _new_cart_line_id() -> str:
    return uuid4().hex


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    context = Column(String, default="")

class CartItem(Base):
    __tablename__ = "cart_items"
    id = Column(Integer, primary_key=True, index=True)
    cart_line_id = Column(
        String,
        default=_new_cart_line_id,
        nullable=False,
        unique=True,
        index=True,
    )
    user_id = Column(Integer, index=True)
    item = Column(String)
    amount = Column(Integer)
    price = Column(Float, nullable=True)


class CartQuantityIdempotency(Base):
    __tablename__ = "cart_quantity_idempotency"
    idempotency_key = Column(String, primary_key=True)
    user_id = Column(Integer, nullable=False)
    cart_line_id = Column(String, nullable=False)
    quantity = Column(Integer, nullable=False)
    response_body = Column(String, nullable=False)


Base.metadata.create_all(bind=engine)


def _ensure_price_column() -> None:
    """Idempotently add the price column for databases created before it existed."""
    with engine.connect() as conn:
        columns = conn.execute(text("PRAGMA table_info(cart_items)")).fetchall()
        if not any(col[1] == "price" for col in columns):
            try:
                conn.execute(text("ALTER TABLE cart_items ADD COLUMN price REAL"))
                conn.commit()
                logging.info("memory-retriever | added price column to cart_items")
            except Exception as exc:
                logging.warning(f"memory-retriever | could not add price column: {exc}")


def _ensure_cart_line_id_column() -> None:
    """Add and backfill opaque cart-line IDs for existing SQLite databases."""

    with engine.begin() as conn:
        columns = conn.execute(text("PRAGMA table_info(cart_items)")).fetchall()
        if not any(col[1] == "cart_line_id" for col in columns):
            conn.execute(text("ALTER TABLE cart_items ADD COLUMN cart_line_id TEXT"))
        rows = conn.execute(
            text(
                "SELECT id FROM cart_items "
                "WHERE cart_line_id IS NULL OR cart_line_id = ''"
            )
        ).fetchall()
        for row in rows:
            conn.execute(
                text(
                    "UPDATE cart_items SET cart_line_id = :cart_line_id "
                    "WHERE id = :id"
                ),
                {"cart_line_id": _new_cart_line_id(), "id": row[0]},
            )
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ix_cart_items_cart_line_id "
                "ON cart_items (cart_line_id)"
            )
        )


_ensure_price_column()
_ensure_cart_line_id_column()


class ContextUpdate(BaseModel):
    new_context: str

class ItemUpdate(BaseModel):
    item: str
    amount: int
    price: Optional[float] = None

class CartQuantityUpdate(BaseModel):
    quantity: int = Field(ge=0)
    idempotency_key: str = Field(..., min_length=1)

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _cart_item_dict(item: CartItem) -> dict:
    return {
        "cart_line_id": item.cart_line_id,
        "item": item.item,
        "amount": item.amount,
        "price": item.price,
    }


def _replay_cart_quantity_update(
    db,
    idempotency_key: str,
    user_id: int,
    cart_line_id: str,
    quantity: int,
) -> dict | None:
    record = db.query(CartQuantityIdempotency).filter(
        CartQuantityIdempotency.idempotency_key == idempotency_key
    ).first()
    if record is None:
        return None
    if (
        record.user_id != user_id
        or record.cart_line_id != cart_line_id
        or record.quantity != quantity
    ):
        raise HTTPException(
            status_code=409,
            detail="Idempotency key was already used for a different cart mutation",
        )
    return json.loads(record.response_body)


@app.get("/user/{user_id}")
async def get_user(user_id: int, db=Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    cart_items = db.query(CartItem).filter(CartItem.id == user_id).all()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": user.id, "context": user.context, "cart": [_cart_item_dict(item) for item in cart_items]}

@app.get("/user/{user_id}/cart")
async def report_cart(user_id: int, db=Depends(get_db)):
    cart_items = db.query(CartItem).filter(CartItem.user_id == user_id).all()
    if not cart_items:
        return {
            "user_id": user_id,
            "cart": []
        }      
    else:
        return {
            "user_id": user_id,
            "cart": [_cart_item_dict(item) for item in cart_items]
        }
  
@app.get("/user/{user_id}/context")
async def get_context(user_id: int, db=Depends(get_db)):
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
async def add_to_cart(
    user_id: int,
    item_update: ItemUpdate,
    db=Depends(get_db),
):
    item = item_update.item
    amount = item_update.amount
    price = item_update.price
    cart_item = db.query(CartItem).filter(CartItem.user_id == user_id, CartItem.item == item).first()
    if cart_item:
        cart_item.amount += amount
        # Refresh price if the caller provides a newer value; keep existing otherwise.
        if price is not None:
            cart_item.price = price
    else:
        cart_item = CartItem(user_id=user_id, item=item, amount=amount, price=price)
        db.add(cart_item)
    db.commit()
    return {
        "user_id": user_id,
        "message": f"In response to the user's request, I have added {amount} of '{item}' to their cart."
        }

@app.post("/user/{user_id}/cart/remove")
async def remove_cart(
    user_id: int,
    item_update: ItemUpdate,
    db=Depends(get_db),
):
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

@app.put("/user/{user_id}/cart/{cart_line_id}/quantity")
async def update_cart_quantity(
    user_id: int,
    cart_line_id: str,
    quantity_update: CartQuantityUpdate,
    db=Depends(get_db),
):
    try:
        replay = _replay_cart_quantity_update(
            db,
            quantity_update.idempotency_key,
            user_id,
            cart_line_id,
            quantity_update.quantity,
        )
        if replay is not None:
            return replay
        cart_item = db.query(CartItem).filter(
            CartItem.cart_line_id == cart_line_id,
            CartItem.user_id == user_id,
        ).first()
        if not cart_item:
            raise HTTPException(status_code=404, detail="Cart line not found")
        item = cart_item.item
        cart_line = _cart_item_dict(cart_item)
        cart_line["amount"] = quantity_update.quantity
        if quantity_update.quantity == 0:
            db.delete(cart_item)
        else:
            cart_item.amount = quantity_update.quantity
        response = {
            "user_id": user_id,
            "cart_line": cart_line,
            "message": f"Updated '{item}' to quantity {quantity_update.quantity}.",
        }
        db.add(
            CartQuantityIdempotency(
                idempotency_key=quantity_update.idempotency_key,
                user_id=user_id,
                cart_line_id=cart_line_id,
                quantity=quantity_update.quantity,
                response_body=json.dumps(response),
            )
        )
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            replay = _replay_cart_quantity_update(
                db,
                quantity_update.idempotency_key,
                user_id,
                cart_line_id,
                quantity_update.quantity,
            )
            if replay is None:
                raise
            return replay
        return response
    except Exception:
        db.rollback()
        raise

@app.post("/user/{user_id}/cart/clear")
async def clear_cart(user_id: int, db=Depends(get_db)):
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
async def add_context(
    user_id: int,
    context_update: ContextUpdate,
    db=Depends(get_db),
):
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
async def replace_context(
    user_id: int,
    context_update: ContextUpdate,
    db=Depends(get_db),
):
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
async def clear_context(user_id: int, db=Depends(get_db)):
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
async def clear_user(user_id: int, db=Depends(get_db)):
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
