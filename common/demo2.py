from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Union


class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

app = FastAPI()


@app.get("/items")
async def read_item(q: Union[str, None] = Query(default=None, max_length=50)):
    """
    Read an item with the given ID and optional query parameter.
    """
    results = {"items": [{"item_id": "Foo"},{"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results

@app.post("/items/")
async def create_item(item: Item):
    """
    Create an item with the given name, description, price, and tax.
    """
    item_dict = item.model_dump()
    price_with_tax = item.price + (item.tax or 0.0)
    item_dict.update({"price_with_tax": price_with_tax})
    return item_dict

@app.put("/items/{item_id}")
@app.post("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    """
    Update an item with the given ID.
    """
    return {"item_id": item_id, **item.model_dump()}
