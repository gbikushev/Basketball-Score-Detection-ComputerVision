from fastapi import File, HTTPException, UploadFile, APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy import select, insert, delete
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
import os
import re
import uuid
import threading
import asyncio
from fastapi.responses import JSONResponse
from src.process_video.functions import video_process
from src.process_video.video_func import check_file_format, prepare_video_files, calculate_video_size, change_fps
from src.database import get_async_session
from src.process_video.models import video
from src.auth.models import User
from src.auth.models import user as user_table
from src.auth.base_config import current_user, get_jwt_strategy
from src.process_video.connection_manager import ConnectionManager  # Import the ConnectionManager

from converter import Converter

manager = ConnectionManager()
router = APIRouter(prefix="/videos", 
                   tags=["video"])

# Global dictionary to store results
results_storage = {}

# Global dictionary to hold stop events
stop_events = {}


@router.get("/statistics/{throw_type}")
async def stat_all_video(throw_type: str, session: AsyncSession = Depends(get_async_session),
                         user: User = Depends(current_user)):
    query = select(video).where(video.c.user_id == user.id)
    result = await session.execute(query)
    videos = result.mappings().all()

    if throw_type == 'all':
        videos_filtered = [{key: value for key, value in d.items()} for d in videos]
    else:
        videos_filtered = [{key: value for key, value in d.items()} for d in videos if d.get('throw_type') == throw_type]

    return videos_filtered


@router.post("/upload_video_endpoint/{throw_type}")
async def upload_video(throw_type: str, file: UploadFile = File(...), session: AsyncSession = Depends(get_async_session),
                       user: User = Depends(current_user)):

    if not check_file_format(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file format. Acceptable format .mp4")
    
    video_path, processed_video_path = await prepare_video_files(file)
    video_size = await calculate_video_size(video_path)

    # Extend token lifetime during processing
    extended_jwt = get_jwt_strategy(processing=True)
    access_token = await extended_jwt.write_token(user)
    response = JSONResponse(content={"success": True, "message": "Video processing started"})

    # Set the extended JWT token in the cookie
    response.set_cookie(key="bonds", value=access_token, httponly=True, max_age=7200)

    # Create a dictionary to hold the results
    result_holder = {
        'video_name': file.filename,
        'video_size': video_size,
        'video_path': video_path,
        'throw_type': throw_type,
        'processed_video_path': processed_video_path
    }

    task_id = str(uuid.uuid4())
    results_storage[task_id] = result_holder

    # Create a stop event for the thread
    stop_event = threading.Event()
    stop_events[task_id] = stop_event

    loop = asyncio.get_event_loop()
    thread = threading.Thread(target=video_process, args=(video_path, processed_video_path, manager, loop, result_holder, stop_event))
    thread.start()

    return JSONResponse(content={"success": True, "message": "Video processing started", "task_id": task_id})


@router.get("/get_video_processing_results/{task_id}")
async def get_video_processing_results(task_id: str, session: AsyncSession = Depends(get_async_session),
                                       user: User = Depends(current_user)):
    if task_id not in results_storage:
        raise HTTPException(status_code=404, detail="Task ID not found")

    result_holder = results_storage[task_id]

    if not result_holder:
        return JSONResponse(content={"success": False, "message": "The results of the video processing are not obtained"})

    # Check if the processing has been completed
    if "score" not in result_holder or "misses" not in result_holder or "throws" not in result_holder:
        return JSONResponse(content={"success": False, "message": "Video processing is not yet completed"})

    # Retrieve the results
    video_name = result_holder.get('video_name')
    video_size = result_holder.get('video_size')
    score = result_holder.get('score')
    misses = result_holder.get('misses')
    throws = result_holder.get('throws')
    throw_type = result_holder.get('throw_type')
    processed_video_path = result_holder.get('processed_video_path')

    # remove the input video from the directory with result videos
    video_path = result_holder.get('video_path')
    os.remove(video_path)

    # change fps of result video
    output_path = change_fps(processed_video_path)
    os.remove(processed_video_path)


    try:
        accuracy = round(100 * score / throws)
    except ZeroDivisionError:
        accuracy = 0

    query = select(user_table.c.id).where(user_table.c.username == user.username)
    result = await session.execute(query)
    user_id = result.scalars().all()[0]

    stmt = insert(video).values(name_video=video_name, video_size=video_size,
                                number_throws=throws, number_goals=score,
                                number_misses=misses, accuracy=accuracy, throw_type=throw_type,
                                date_upload=datetime.utcnow(), user_id=user_id, video_path=output_path)

    await session.execute(stmt)
    await session.commit()

    # Clear results after retrieving
    results_storage.pop(task_id, None)

    return JSONResponse(content={"success": True, "message": "Video processing completed", "score": score, "misses": misses, "throws": throws})


@router.delete("/delete_video/{video_path}")
async def delete_video(video_path: str, session: AsyncSession = Depends(get_async_session),
                       user: User = Depends(current_user)):
    
    
    # Fix the the video_path
    video_path = video_path.replace("result_videosesult", "result_videos\\result", 1)

    # Define the DELETE statement
    stmt = delete(video).where(video.c.video_path == video_path)
    
    # # Execute the DELETE statement
    result = await session.execute(stmt)

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Video not found")

    await session.commit()

    os.remove(video_path)

    return {"success": True}
        
# Endpoint to stop the threads
@router.post("/stop_video_processing/{task_id}")
async def stop_video_processing(task_id: str):
    if task_id in stop_events:
        stop_events[task_id].set()
        return JSONResponse(content={"success": True, "message": "Video processing stopped"})
    else:
        return JSONResponse(content={"success": False, "message": "Invalid task_id"})


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
