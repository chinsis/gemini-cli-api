from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Union, Annotated


class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

app = FastAPI()


@app.get("/items")
# async def read_item(q: Union[str, None] = Query(default=None, max_length=50)):
async def read_item(q: Annotated[str | None, Query(max_length=50)] = None):
    """
    Union[str, None] = Query(default=None, max_length=50) 类型声明和规则校验混在一起
    Annotated[str | None, Query(max_length=50)] = None 类型声明和规则校验分离
    """
    results = {"items": [{"item_id": "Foo"},{"item_id": "Bar"}]}
    if q:
        # pylance warning:pylance根据resulets推断键值对应该为str：[], update更新的是一个字符串，所以有了错误提示，实际不影响项目运行
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
