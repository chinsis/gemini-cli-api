from typing import Annotated, Any
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field, HttpUrl

app = FastAPI()

class Image(BaseModel):
    url: HttpUrl
    name: str | None = Field(default=None, title="The name of the image", max_length=100)

class Item(BaseModel):
    name: str
    description: str | None = Field(default=None, title="The description of the item", max_length=300)
    price: float
    tax: float | None = Field(default=None, title="must greater than zero ", gt=0)
    tags: set[Any] = set()
    image: list[Image] | None = None

class Offer(BaseModel):
    name: str
    description: str | None = Field(default=None, title="The description of the offer", max_length=300)
    price: float
    tax: float | None = Field(default=None, title="must greater than zero ", gt=0)
    tags: set[Any] = set()
    item: list[Item] | None = None

@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Annotated[Item, Body()]):
    return {"item_id": item_id, **item.model_dump()}

@app.post("/offers/")
async def create_offer(offer: Annotated[Offer, Body()]):
    return offer

@app.post("/images/multiple/")
async def create_multiple_images(images: list[Image]):
    return images