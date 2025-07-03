from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./context.db"
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
def get_user(user_id: int):
    db = SessionLocal()
    user = db.query(User).filter(User.id == user_id).first()
    cart_items = db.query(CartItem).filter(CartItem.id == user_id).all()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": user.id, "context": user.context, "cart": [{"item": item.item, "amount": item.amount} for item in cart_items]}

@app.get("/user/{user_id}/cart")
def report_cart(user_id: int):
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
def get_context(user_id: int):
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
def add_to_cart(user_id: int, item_update: ItemUpdate):
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
        "message": f"Added {amount} of '{item}' to cart"
        }

@app.post("/user/{user_id}/cart/remove")
def remove_cart(user_id: int, item_update: ItemUpdate):
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
        "message": f"Removed {amount} of '{item}' from cart"
        }

@app.post("/user/{user_id}/cart/clear")
def clear_cart(user_id: int):
    db = SessionLocal()
    cart_items = db.query(CartItem).filter(CartItem.user_id == user_id).all()
    if not cart_items:
        raise HTTPException(status_code=404, detail="No items found in cart")
    for item in cart_items:
        db.delete(item)
    db.commit()
    return {
        "user_id": user_id,
        "message": f"Cart for user {user_id} has been deleted."
        }

@app.post("/user/{user_id}/context/add")
def add_context(user_id: int, context_update: ContextUpdate):
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
def replace_context(user_id: int, context_update: ContextUpdate):
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
def clear_context(user_id: int):
    db = SessionLocal()
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    return {
        "user_id": user_id,
        "message": f"Context for user {user_id} has been deleted."
        }

@app.post("/user/{user_id}/clear")
def clear_user(user_id: int):
    db = SessionLocal()
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    return {
        "user_id": user_id,
        "message": f"Deleted cart and context for user {user_id}"
        }
