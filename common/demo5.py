from datetime import datetime, time, timedelta
from typing import Annotated
from uuid import UUID
from fastapi import FastAPI, Body


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