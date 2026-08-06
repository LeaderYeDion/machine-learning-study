import json
from typing import Annotated

from fastapi import FastAPI, Query
from fastapi.params import Cookie, Header, Depends
from pydantic import BaseModel, HttpUrl

app = FastAPI()


class User(BaseModel):
    id: int
    username: str
    email: str


class MyQuery(BaseModel):
    username: str
    email: str


class Image(BaseModel):
    url: HttpUrl
    name: str


class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None
    tags: set[str] = set()
    images: list[Image] | None = None


class Cookies(BaseModel):
    model_config = {"extra": "forbid"}

    session_id: str
    fatebook_tracker: str | None = None
    googall_tracker: str | None = None


class CommonHeaders(BaseModel):
    model_config = {"extra": "forbid"}

    host: str
    save_data: bool
    if_modified_since: str | None = None
    traceparent: str | None = None
    x_tag: list[str] = []


@app.get("/")
async def root(q: Annotated[list[str] | None, Query()] = None):
    return {"message": f"""Hello World from {q}"""}


@app.get("/hello")
async def hello(name: str, query: MyQuery | None = None) -> User:
    """
    :param name:
    :param query: query对应类型是pydantic,默认会被当成request body里的内容解析
    :return:
    """
    user = User(id=123, username=name, email="2232660905@qq.com")
    return user


@app.get("/world")
async def helloWorld(name: str, query: Annotated[MyQuery, Query(max_length=50)]) -> User:
    """
    :param name:
    :param query: query对应类型是pydantic,默认会被当成request body里的内容解析
    :return:
    """
    user = User(id=123, username=name, email="2232660905@qq.com")
    return user


@app.get("/items/")
async def get_item_desc(item: Item, cookie: Annotated[Cookies | None, Cookie()],
                        headers: Annotated[CommonHeaders | None, Header()]) -> str:
    return f"""{item} + \n + {cookie} +\n+ {headers}"""

class CommonQueryParams:
    def __init__(self, q: str | None = None, skip: int = 0, limit: int = 100):
        self.q = q
        self.skip = skip
        self.limit = limit


@app.get("/items/")
async def read_items(commons: Annotated[CommonQueryParams, Depends()]) -> str:
    return json.dumps(commons)

async def common_parameters(q: str | None = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/users/")
async def read_users(commons: Annotated[dict, Depends(common_parameters)]):
    return commons
