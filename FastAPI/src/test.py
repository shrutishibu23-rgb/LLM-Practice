from fastapi import FastAPI

test = FastAPI()

@test.get("/hello-world")
def helloWorld():
    return {"message": "Hello, world!"}