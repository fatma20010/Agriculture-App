from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status, Security, Request
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
import logging

# Database setup
DATABASE_URL = "sqlite:///./users.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Authentication settings
SECRET_KEY = "YOUR_SECRET_KEY_CHANGE_THIS_IN_PRODUCTION"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Database Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String, nullable=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    analysis_records = relationship("AnalysisRecord", back_populates="user")

class AnalysisRecord(Base):
    __tablename__ = "analysis_records"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    analysis_type = Column(String)
    result_data = Column(Text)  # JSON string of results
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="analysis_records")

# Create tables
Base.metadata.create_all(bind=engine)

# Authentication models
class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: int
    username: str
    is_admin: bool

class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: List[str] = []

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
    is_admin: bool = False

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    is_admin: bool
    is_active: bool
    created_at: datetime
    class Config:
        from_attributes = True

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Dependency to get current user (dummy, replace with your real logic)
def get_current_user(request: Request, db: Session = Depends(get_db)):
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token: no subject")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user
