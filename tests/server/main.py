from fastapi import FastAPI
from survos2.start_server import app


# app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Tomato"}
