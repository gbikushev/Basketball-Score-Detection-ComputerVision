from fastapi import WebSocket
from typing import List

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

    async def send_progress(self, progress: float):
        for connection in self.active_connections:
            await connection.send_json({"progress": progress})

    async def send_completion_message(self):
        for connection in self.active_connections:
            await connection.send_json({"message": "Video uploaded successfully"})
        for connection in self.active_connections:
            await connection.close()