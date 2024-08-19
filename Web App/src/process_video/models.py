from sqlalchemy import Table, Column, Integer, String, TIMESTAMP, MetaData, ForeignKey
from src.auth.models import user
from datetime import datetime

metadata = MetaData()

video = Table(
    "video",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name_video", String, nullable=False),
    Column("video_size", Integer, nullable=False),
    Column("number_throws", Integer, nullable=False),
    Column("number_goals", Integer, nullable=False),
    Column("number_misses", Integer, nullable=False),
    Column("accuracy", Integer, nullable=False),
    Column("throw_type", String, nullable=False),
    Column("date_upload", TIMESTAMP, default=datetime.utcnow),
    Column("user_id", Integer, ForeignKey(user.c.id)),
    Column("video_path", String, nullable=False),  # New column for video path
)
