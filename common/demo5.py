from datetime import datetime, time, timedelta
from typing import Annotated
from uuid import UUID
from fastapi import FastAPI, Body, Cookie, Header


app = FastAPI()

@app.put("/items/{item_id}")
async def update_item(item_id: UUID, 
                      start_time: Annotated[datetime, Body()],
                      end_time: Annotated[datetime, Body()],
                      process_after: Annotated[timedelta, Body()],
                      repeat_at: Annotated[time | None, Body()]=None):
    """
    Update an item with the given item_id.
    The item is a dictionary with keys 'name', 'description', 'price', and 'tax'.
    """
    start_process = start_time + process_after
    duration = end_time - start_process
    return {
        "item_id": item_id,
        "start_datetime": start_time,
        "end_datetime": end_time,
        "process_after": process_after,
        "repeat_at": repeat_at,
        "start_process": start_process,
        "duration": duration,
    }

# @app.get("/items/")
# async def read_items(useragent: Annotated[str | None, Header()] = None,
#                      session_id: Annotated[str | None, Cookie()] = None):
#     """
#     Read items with the given parameters.
#     """
#     return {
#         "User_Agent": useragent,
#         "Cookie": session_id
#     }

@app.get("/items/")
async def read_items(
    user_agent: Annotated[str | None, Header()] = None,
    session_id: Annotated[str | None, Cookie()] = None
):
    return {
        "User_Agent": user_agent,
        "Cookie": session_id
    }