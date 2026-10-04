import asyncio
import json
import logging
import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mock_assemblyai")

async def handler(websocket):
    logger.info("Client connected to mock AssemblyAI")
    
    try:
        # Send SessionBegins
        await websocket.send(json.dumps({"message_type": "SessionBegins"}))
        logger.info("Sent SessionBegins")
        
        # Start a background task to consume audio
        async def consume_audio():
            try:
                async for message in websocket:
                    pass
            except websockets.exceptions.ConnectionClosed:
                pass
        consumer_task = asyncio.create_task(consume_audio())
        
        # Stream the requested test phrase
        await asyncio.sleep(1.0)
        await websocket.send(json.dumps({
            "message_type": "PartialTranscript",
            "text": "I need to plan a customer workshop in Pune for around thirty people.",
            "audio_start": 1000
        }))
        logger.info("Sent Partial 1")
        
        await asyncio.sleep(1.5)
        await websocket.send(json.dumps({
            "message_type": "PartialTranscript",
            "text": "I need to plan a customer workshop in Pune for around thirty people. I'm looking for a suitable venue",
            "audio_start": 2500
        }))
        logger.info("Sent Partial 2")
        
        await asyncio.sleep(1.5)
        await websocket.send(json.dumps({
            "message_type": "FinalTranscript",
            "text": "I need to plan a customer workshop in Pune for around thirty people. I'm looking for a suitable venue, and I also need to know the cancellation policy and whether catering is available.",
            "audio_start": 4000
        }))
        logger.info("Sent Final")
        
        await asyncio.sleep(0.5)
        
        consumer_task.cancel()
    except Exception as e:
        logger.error(f"Error: {e}")

async def main():
    async with websockets.serve(handler, "localhost", 8765):
        logger.info("Mock AssemblyAI server listening on ws://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
