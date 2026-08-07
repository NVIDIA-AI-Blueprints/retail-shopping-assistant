# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import asynccontextmanager
import json
import time

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from typing import Optional

from .conversations import create_conversation_router, sweep_abandoned_turns
from .database import Base, DATABASE_URL, SessionLocal, build_engine, engine
from .migrations import (
    cart_mutation_digest,
    ensure_cart_line_id_column,
    ensure_price_column,
    ensure_product_id_column,
    migrate_quantity_idempotency,
    run_schema_migrations,
)
from .models import (
    CartItem,
    CartMutation,
    CartQuantityIdempotency,
    ConversationEvent,
    ConversationProjection,
    ConversationTurn,
    SchemaMigration,
    ShopperProfile,
    User,
    new_cart_line_id,
)
from .shopper_profiles import (
    bootstrap_shopper_profiles,
    create_shopper_profile_router,
)


__all__ = (
    "Base",
    "CartItem",
    "CartMutation",
    "CartQuantityIdempotency",
    "ConversationEvent",
    "ConversationProjection",
    "ConversationTurn",
    "DATABASE_URL",
    "SchemaMigration",
    "ShopperProfile",
    "User",
    "build_engine",
)


def _new_cart_line_id() -> str:
    return new_cart_line_id()


def _ensure_price_column() -> None:
    """Idempotently add the price column for databases created before it existed."""
    with engine.begin() as connection:
        ensure_price_column(connection)


def _ensure_cart_line_id_column() -> None:
    """Add and backfill opaque cart-line IDs for existing SQLite databases."""
    with engine.begin() as connection:
        ensure_cart_line_id_column(connection)


def _ensure_product_id_column() -> None:
    """Idempotently add catalog product identity to existing cart rows."""
    with engine.begin() as connection:
        ensure_product_id_column(connection)


def _cart_mutation_digest(
    operation: str,
    stable_target_id: str,
    request_body: dict,
) -> str:
    return cart_mutation_digest(operation, stable_target_id, request_body)


def _migrate_quantity_idempotency() -> None:
    """Copy existing quantity replay records into the unified ledger once."""
    with engine.begin() as connection:
        migrate_quantity_idempotency(connection)


def _run_schema_migrations() -> None:
    run_schema_migrations(engine)


def _bootstrap_shopper_profiles(
    *,
    seed_path: str | None = None,
) -> None:
    bootstrap_shopper_profiles(SessionLocal, seed_path=seed_path)


def _sweep_abandoned_turns(
    *,
    now: float | None = None,
    timeout_seconds: int | None = None,
) -> int:
    return sweep_abandoned_turns(
        SessionLocal,
        now=now,
        timeout_seconds=timeout_seconds,
    )


class ContextUpdate(BaseModel):
    new_context: str

class ItemUpdate(BaseModel):
    item: str
    amount: int = Field(gt=0)
    price: Optional[float] = None
    product_id: str = Field(..., min_length=1)
    idempotency_key: str = Field(..., min_length=1)
    #: None for one-size goods. Reaches both the merge key and the idempotency
    #: digest below, because two sizes of one product are two different things
    #: a shopper owns and two different mutations.
    size: Optional[str] = Field(default=None, max_length=32)

class CartRemoveUpdate(BaseModel):
    amount: int = Field(gt=0)
    cart_line_id: str = Field(..., min_length=1)
    idempotency_key: str = Field(..., min_length=1)

class CartQuantityUpdate(BaseModel):
    quantity: int = Field(ge=0)
    idempotency_key: str = Field(..., min_length=1)

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _run_schema_migrations()
    _bootstrap_shopper_profiles()
    _sweep_abandoned_turns()
    yield


app = FastAPI(lifespan=_lifespan)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app.include_router(create_conversation_router(get_db))
app.include_router(create_shopper_profile_router(get_db))


def _cart_item_dict(item: CartItem) -> dict:
    result = {
        "cart_line_id": item.cart_line_id,
        "item": item.item,
        "amount": item.amount,
        "price": item.price,
    }
    if item.product_id:
        result["product_id"] = item.product_id
    if item.size:
        result["size"] = item.size
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


def _cart_item_for_add(
    db, user_id: int, product_id: str, size: str | None = None
) -> CartItem | None:
    """Find the line this add should merge into.

    Keyed on size as well as product: a 6 and an 8 of one dress are two things
    the shopper owns, not one line of quantity two.
    """

    return db.query(CartItem).filter(
        CartItem.user_id == user_id,
        CartItem.product_id == product_id,
        CartItem.size.is_(None) if size is None else CartItem.size == size,
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
    size = item_update.size
    canonical_digest = _cart_mutation_digest(
        "add",
        stable_target_id,
        # Size belongs in the digest, or adding a 6 and then an 8 replays as
        # one mutation and the second silently vanishes.
        {"amount": amount, "item": item, "price": price, "size": size},
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
        cart_item = _cart_item_for_add(db, user_id, item_update.product_id, size)
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
                size=size,
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
