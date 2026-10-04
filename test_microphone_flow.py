import asyncio
import websockets
import httpx
import json
import logging
from httpx_sse import aconnect_sse

logging.basicConfig(level=logging.INFO)

async def test_flow():
    # 1. Create Session
    async with httpx.AsyncClient() as client:
        res = await client.post("http://localhost:8000/api/session")
        session_id = res.json()["session_id"]
        logging.info(f"Created session: {session_id}")
    
    # 2. Listen to SSE Stream in background
    async def listen_sse():
        try:
            async with httpx.AsyncClient() as client:
                async with aconnect_sse(client, "GET", f"http://localhost:8000/api/session/{session_id}/stream") as event_source:
                    async for sse in event_source.aiter_sse():
                        logging.info(f"SSE Event: {sse.event} | Data: {sse.data}")
        except Exception as e:
            logging.info(f"SSE loop ended: {e}")

    sse_task = asyncio.create_task(listen_sse())

    # 3. Connect to WebSocket
    uri = f"ws://localhost:8000/api/session/{session_id}/stt/audio"
    async with websockets.connect(uri) as ws:
        logging.info("Connected to backend STT WebSocket")
        await ws.send(b"\x00" * 4096)
        logging.info("Sent dummy PCM bytes")
        
        await asyncio.sleep(5)
        
        try:
            await ws.send("STOP")
            logging.info("Sent STOP")
        except Exception as e:
            logging.info(f"Could not send STOP: {e}")

    await asyncio.sleep(1)
    sse_task.cancel()

if __name__ == "__main__":
    asyncio.run(test_flow())

