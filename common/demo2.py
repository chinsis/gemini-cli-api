from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Union, Annotated, Any


class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

app = FastAPI()


@app.get("/items")
# async def read_item(q: Union[str, None] = Query(default=None, max_length=50)):
async def read_item(q: Annotated[str, Query(max_length=50, min_length=3, pattern="^[A-Za-z]+$", alias="item-query", deprecated=True)]="findindex"):
    """
    Union[str, None] = Query(default=None, max_length=50) 类型声明和规则校验混在一起
    Annotated[str | None, Query(max_length=50)] = None 类型声明和规则校验分离
    使用Annotated[]="findindex"设置default值，不能使用Annotated[Query(default="findindex")]，两者有冲突
    Query(pattern="^[A-Za-z]+$", regex="" ) 设置正则表达式校验，其中regex为pydantic v1的正则表达式校验方式，pattern为pydantic v2的正则表达式校验方式，v1已弃用，推荐v2
    alias="item-query" 设置别名，FastAPI会在文档中显示该别名
    Deprecated=True 设置该参数为弃用状态，FastAPI会在文档中标记该参数为弃用
    """
    results: dict[str, Any] = {"items": [{"item_id": "Foo"},{"item_id": "Bar"}]}

    if q:
        # pylance warning:pylance根据resulets推断键值对应该为str：[], update更新的是一个字符串，所以有了错误提示，实际不影响项目运行
        # 如果result声明了类型为dict[str, Any]，则不会再有报错提示
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
