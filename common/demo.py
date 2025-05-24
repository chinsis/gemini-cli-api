from fastapi import FastAPI
from enum import Enum
from pydantic import BaseModel

app = FastAPI()

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

fake_items_db = [
    {"item_name": "Foo"},
    {"item_name": "Bar"},  
    {"item_name": "Baz"},
]
@app.get("/")
async def root():
    return {"message": "Hello World"}

# @app.get("/items/{item_id}")
# async def read_item(item_id: int):
#     return {"item_id": item_id}

@app.get("/users/me")
def read_user_me():
    return {"user": "me"}

@app.get("/users/{user_id}")
def read_user(user_id: int):
    return {"user": user_id} 

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}

@app.get("/items")
async def itmes(skip: int = 0, limit: int = 10):
    # 返回切片列表
    return fake_items_db[skip : skip + limit]

# @app.get("/items/{item_id}")
# async def read_item(item_id: str, q: str | None=None, short: bool=False):
#     item = {"item_id": item_id}
#     """
#     q是可选参数，可以是string,也可以是None，默认值是None
#     short是bool类型的可选参数，默认值是False，fastapi会自动将其转换为bool类型
#     """
#     if q:
#         item.update({"q": q})
#     if not short:
#         item.update({
#             "description": "This is an amazing item that has a long description"
#             })
#     return item

@app.get("/users/{user_id}/items/{item_id}")
def read_user_item(user_id: int, item_id: str, q: str | None = None, short: bool = False):
    item = {"owner_id":user_id, "item_id": item_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update(
            {"description": "This is an amazing item that has a long description"}
        )
    return item

@app.get("/items/{item_id}")
async def read_user_item(item_id: str, needy: str, skip: int = 0, limit: int | None=None):
    item = {"item_id": item_id, "needy": needy}
    return item[skip : skip + limit] if limit else item
