from fastapi import FastAPI
from app.api.api import router

app = FastAPI(title="StackBot API")

app.include_router(router, prefix="/api")
