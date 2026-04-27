from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Depends
from src.schemas import PostCreate, PostResponse, UserCreate, UserRead, UserUpdate
from src.models import Post, User
from contextlib import asynccontextmanager
from .database import get_db
from sqlalchemy import select
from sqlalchemy.orm import Session
from src.images import imagekit
from imagekitio.models.UploadFileRequestOptions import UploadFileRequestOptions
from src.auth import hash_password, create_access_token, verify_password, get_current_user

import shutil
import os
import uuid
import tempfile #We will create a temp file first when the user uploads an image and once it is added to the DB we will delete it

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    print("App starting up")
    yield
    
    # shutdown
    print("App shutting down")

app = FastAPI(lifespan=lifespan)

@app.post("/register")
def register(email: str, password: str, db: Session = Depends(get_db)):
    try:
        existing_user = db.query(User).filter(User.email == email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        hashed_password = hash_password(password)

        new_user = User(
            email=email,
            hashed_password=hashed_password,
            is_active=True
        )

        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        print("USER CREATED:", new_user.email)

        return {"message": "User created successfully"}

    except Exception as e:
        print("REGISTER ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/login")
def login(email: str, password: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()

    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    token = create_access_token(data={"sub": user.id})

    return {"access_token": token, "token_type": "bearer"}

@app.post("/upload")
def upload_file(
    file: UploadFile = File(...),
    caption: str = Form(""),
    session: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    temp_file_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)
            
        upload_result = imagekit.upload_file(
            file = open(temp_file_path, "rb"),
            file_name = file.filename,
            options = UploadFileRequestOptions(
                use_unique_file_name = True,
                tags = ["backend-upload"]
            )
        )  
        
        if upload_result.response_metadata.http_status_code == 200:   
            post = Post(
                user_id = current_user.id,
                caption = caption,
                url = upload_result.url, 
                filetype = "video" if file.content_type.startswith("video/") else "image",
                filename = upload_result.name
            )
            session.add(post)
            session.commit()
            session.refresh(post)
            
            return post
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        file.file.close()

@app.get("/feed")
def get_feed(
    session: Session = Depends(get_db)
):
    result = session.execute(select(Post).order_by(Post.created_at.desc()))
    posts = [row[0] for row in result.all()]
    
    posts_data = []
    for post in posts:
        posts_data.append(
            {
                "id": str(post.id),
                "caption": post.caption,
                "url": post.url,
                "filetype": post.filetype,
                "filename": post.filename,
                "created_at": post.created_at.isoformat()
            }
        )
    
    return {"posts": posts_data}

@app.delete("/posts/{post_id}")
def delete_post(
    post_id: str, 
    session: Session = Depends(get_db)
):
    try:
        post_uuid = uuid.UUID(post_id)
        result = session.execute(select(Post).where(Post.id == post_uuid))
        post = result.scalars().first()
        
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        
        session.delete(post)
        session.commit()
        
        return {"success":True, "message": "Post deleted"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""
INITIAL TESTING WITHOUT DATABASE

textPosts = {}

@app.get("/posts")
def getAllPosts(limit: int = None): #Remove "=None" if parameter is mandatory
    if limit:
        #If limit is specified, then return textPosts upto limit
        return list(textPosts.values())[:limit]
    return textPosts

@app.get("/posts/{id}") 
def getPost(id: int) -> PostResponse:
    if id not in textPosts:
        raise HTTPException(status_code= 404, detail="ID not found")
    return textPosts.get(id)

@app.post("/posts")
def createPost(post: PostCreate) -> PostResponse:
    newPost = {"title": post.title, "content": post.content}
    textPosts[max(textPosts) + 1] = newPost
    return newPost
    
"""
