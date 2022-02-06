from typing import Optional
from fastapi import FastAPI
from enum import Enum
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None


fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]

@app.get("/users/{user_id}/models/{model_name}")
async def get_model(model_name: ModelName, user_id: str, skip: Optional[int] = 0, limit: Optional[int] = 10):
    if user_id != "me":
        return fake_items_db[skip : skip + limit]
    else:
        if model_name == ModelName.alexnet:
            return {"model_name": model_name, "message": "Deep Learning FTW!", "user id":user_id}

        if model_name.value == "lenet":
            return {"model_name": model_name, "message": "LeCNN all the images", "user id":user_id}

        return {"model_name": model_name, "message": "Have some residuals", "user id":user_id}


# @app.post("/items/")
# async def create_item(item: Item):
#     item_dict = item.dict()
#     if item.tax:
#         price_with_tax = item.price + item.tax
#         item_dict.update({"price_with_tax": price_with_tax})
#     return item_dict


@app.put("/items/{item_id}")
async def create_item(item_id: int, item: Item, q: Optional[str] = None):
    result = {"item_id": item_id, **item.dict()}
    if q:
        result.update({"q": q})
    return result

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)