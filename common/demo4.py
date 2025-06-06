from typing import Annotated, Any
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field, HttpUrl

app = FastAPI()

class Item(BaseModel):  
    """
        Query、Path、Body可以使用Field(examples=["Sample Item", "Another Item"]) 给出的示例将会在 OpenAPI 文档中显示，但不会影响实际的请求或响应数据。
        model_config 中的 json_schema_extra 用于提供额外的示例数据，这些数据将会在 OpenAPI 文档中显示。
        例如，以下代码片段将会在 OpenAPI 文档中显示一个示例：
        model_config = {
            "json_schema_extra": {
                "example": {
                    "name": "Sample Item",
                    "description": "This is a sample item",
                    "price": 9.99,
                    "tax": 0.5
                }
            }
        }
        body 中的 examples 参数可以用于提供多个示例数据，这些示例将会在 OpenAPI 文档中显示。
        例如，以下代码片段将会在 OpenAPI 文档中显示一个示例：       
        examples=[
            {
                "name": "Foo",
                "description": "A very nice Item",
                "price": 35.4,
                "tax": 3.2,
            }
        ]
        以上的示例优先级高于 model_config 中的 json_schema_extra，因为它们是直接在 Body 中定义的。model_config 中的示例又优先级于 Field 中的 examples。
    """
    name: str = Field(examples=["Sample Item", "Another Item"], title="The name of the item", max_length=100)
    description: str | None = Field(default=None, title="The description of the item", max_length=300)
    price: float
    tax: float | None = Field(default=None, title="must greater than zero ", gt=0)
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Sample Item",
                "description": "This is a sample item",
                "price": 9.99,
                "tax": 0.5
            }
        }
    }

@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Annotated[Item, Body(
    examples=[
                {
                    "name": "another Item",
                    "description": "A very sss Item",
                    "price": 33.4,
                    "tax": 31.2,
                }
            ],
)]):
    results = {"item_id": item_id, "item": item}
    return results