# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import Column, Float, Integer, String, create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Optional
from hashlib import sha256
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
    product_id = Column(String, nullable=True, index=True)
    item = Column(String)
    amount = Column(Integer)
    price = Column(Float, nullable=True)


class CartQuantityIdempotency(Base):
    """Legacy quantity-only ledger retained as a migration source."""

    __tablename__ = "cart_quantity_idempotency"
    idempotency_key = Column(String, primary_key=True)
    user_id = Column(Integer, nullable=False)
    cart_line_id = Column(String, nullable=False)
    quantity = Column(Integer, nullable=False)
    response_body = Column(String, nullable=False)


class CartMutation(Base):
    __tablename__ = "cart_mutations"
    user_id = Column(Integer, primary_key=True)
    idempotency_key = Column(String, primary_key=True)
    operation = Column(String, nullable=False)
    canonical_digest = Column(String, nullable=False)
    stable_target_id = Column(String, nullable=False)
    response_body = Column(String, nullable=False)


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


def _ensure_product_id_column() -> None:
    """Idempotently add catalog product identity to existing cart rows."""

    with engine.connect() as conn:
        columns = conn.execute(text("PRAGMA table_info(cart_items)")).fetchall()
        if not any(col[1] == "product_id" for col in columns):
            conn.execute(text("ALTER TABLE cart_items ADD COLUMN product_id TEXT"))
            conn.commit()


def _cart_mutation_digest(
    operation: str,
    stable_target_id: str,
    request_body: dict,
) -> str:
    canonical = json.dumps(
        {
            "operation": operation,
            "stable_target_id": stable_target_id,
            "request_body": request_body,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _migrate_quantity_idempotency() -> None:
    """Copy existing quantity replay records into the unified ledger once."""

    with engine.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT idempotency_key, user_id, cart_line_id, quantity, "
                "response_body FROM cart_quantity_idempotency"
            )
        ).mappings()
        for row in rows:
            conn.execute(
                text(
                    "INSERT OR IGNORE INTO cart_mutations "
                    "(user_id, idempotency_key, operation, canonical_digest, "
                    "stable_target_id, response_body) VALUES "
                    "(:user_id, :idempotency_key, 'update', :canonical_digest, "
                    ":stable_target_id, :response_body)"
                ),
                {
                    "user_id": row["user_id"],
                    "idempotency_key": row["idempotency_key"],
                    "canonical_digest": _cart_mutation_digest(
                        "update",
                        row["cart_line_id"],
                        {"quantity": row["quantity"]},
                    ),
                    "stable_target_id": row["cart_line_id"],
                    "response_body": row["response_body"],
                },
            )


Base.metadata.create_all(bind=engine)
_ensure_price_column()
_ensure_cart_line_id_column()
_ensure_product_id_column()
_migrate_quantity_idempotency()


class ContextUpdate(BaseModel):
    new_context: str

class ItemUpdate(BaseModel):
    item: str
    amount: int = Field(gt=0)
    price: Optional[float] = None
    product_id: str = Field(..., min_length=1)
    idempotency_key: str = Field(..., min_length=1)

class CartRemoveUpdate(BaseModel):
    amount: int = Field(gt=0)
    cart_line_id: str = Field(..., min_length=1)
    idempotency_key: str = Field(..., min_length=1)

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
    result = {
        "cart_line_id": item.cart_line_id,
        "item": item.item,
        "amount": item.amount,
        "price": item.price,
    }
    if item.product_id:
        result["product_id"] = item.product_id
    return result


def _replay_cart_mutation(
    db,
    user_id: int,
    idempotency_key: str,
    operation: str,
    stable_target_id: str,
    canonical_digest: str,
) -> dict | None:
    record = db.query(CartMutation).filter(
        CartMutation.user_id == user_id,
        CartMutation.idempotency_key == idempotency_key,
    ).first()
    if record is None:
        return None
    if (
        record.operation != operation
        or record.stable_target_id != stable_target_id
        or record.canonical_digest != canonical_digest
    ):
        raise HTTPException(
            status_code=409,
            detail="Idempotency key was already used for a different cart mutation",
        )
    return json.loads(record.response_body)


def _commit_cart_mutation(
    db,
    *,
    user_id: int,
    idempotency_key: str,
    operation: str,
    stable_target_id: str,
    canonical_digest: str,
    response: dict,
) -> dict:
    db.add(
        CartMutation(
            user_id=user_id,
            idempotency_key=idempotency_key,
            operation=operation,
            stable_target_id=stable_target_id,
            canonical_digest=canonical_digest,
            response_body=json.dumps(response),
        )
    )
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        replay = _replay_cart_mutation(
            db,
            user_id,
            idempotency_key,
            operation,
            stable_target_id,
            canonical_digest,
        )
        if replay is None:
            raise
        return replay
    return response


def _cart_item_for_add(db, user_id: int, product_id: str) -> CartItem | None:
    return db.query(CartItem).filter(
        CartItem.user_id == user_id,
        CartItem.product_id == product_id,
    ).first()


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
    stable_target_id = item_update.product_id
    canonical_digest = _cart_mutation_digest(
        "add",
        stable_target_id,
        {"amount": amount, "item": item, "price": price},
    )
    try:
        replay = _replay_cart_mutation(
            db,
            user_id,
            item_update.idempotency_key,
            "add",
            stable_target_id,
            canonical_digest,
        )
        if replay is not None:
            return replay
        cart_item = _cart_item_for_add(db, user_id, item_update.product_id)
        if cart_item:
            cart_item.amount += amount
            if price is not None:
                cart_item.price = price
        else:
            cart_item = CartItem(
                user_id=user_id,
                product_id=item_update.product_id,
                item=item,
                amount=amount,
                price=price,
            )
            db.add(cart_item)
        db.flush()
        response = {
            "user_id": user_id,
            "cart_line": _cart_item_dict(cart_item),
            "message": (
                f"In response to the user's request, I have added {amount} "
                f"of '{item}' to their cart."
            ),
        }
        return _commit_cart_mutation(
            db,
            user_id=user_id,
            idempotency_key=item_update.idempotency_key,
            operation="add",
            stable_target_id=stable_target_id,
            canonical_digest=canonical_digest,
            response=response,
        )
    except Exception:
        db.rollback()
        raise

@app.post("/user/{user_id}/cart/remove")
async def remove_cart(
    user_id: int,
    item_update: CartRemoveUpdate,
    db=Depends(get_db),
):
    amount = item_update.amount
    stable_target_id = item_update.cart_line_id
    canonical_digest = _cart_mutation_digest(
        "remove",
        stable_target_id,
        {"amount": amount},
    )
    try:
        replay = _replay_cart_mutation(
            db,
            user_id,
            item_update.idempotency_key,
            "remove",
            stable_target_id,
            canonical_digest,
        )
        if replay is not None:
            return replay
        cart_item = db.query(CartItem).filter(
            CartItem.user_id == user_id,
            CartItem.cart_line_id == item_update.cart_line_id,
        ).first()
        if not cart_item:
            raise HTTPException(status_code=404, detail="Item not in cart")
        item = cart_item.item
        remaining = max(0, cart_item.amount - amount)
        cart_line = _cart_item_dict(cart_item)
        cart_line["amount"] = remaining
        if remaining == 0:
            db.delete(cart_item)
        else:
            cart_item.amount = remaining
        response = {
            "user_id": user_id,
            "cart_line": cart_line,
            "message": (
                f"In response to the user's request, I have removed {amount} "
                f"of '{item}' from cart."
            ),
        }
        return _commit_cart_mutation(
            db,
            user_id=user_id,
            idempotency_key=item_update.idempotency_key,
            operation="remove",
            stable_target_id=stable_target_id,
            canonical_digest=canonical_digest,
            response=response,
        )
    except Exception:
        db.rollback()
        raise

@app.put("/user/{user_id}/cart/{cart_line_id}/quantity")
async def update_cart_quantity(
    user_id: int,
    cart_line_id: str,
    quantity_update: CartQuantityUpdate,
    db=Depends(get_db),
):
    canonical_digest = _cart_mutation_digest(
        "update",
        cart_line_id,
        {"quantity": quantity_update.quantity},
    )
    try:
        replay = _replay_cart_mutation(
            db,
            user_id,
            quantity_update.idempotency_key,
            "update",
            cart_line_id,
            canonical_digest,
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
        return _commit_cart_mutation(
            db,
            user_id=user_id,
            idempotency_key=quantity_update.idempotency_key,
            operation="update",
            stable_target_id=cart_line_id,
            canonical_digest=canonical_digest,
            response=response,
        )
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
