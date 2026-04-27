import asyncio
import json
import websockets

WS_URL = "wss://webmessaging.mypurecloud.de/v1?deploymentId=d5cad5f2-e901-4849-8e93-070ad426f8ee"  

payload = {
    "action": "configureSession",
    "deploymentId": "d5cad5f2-e901-4849-8e93-070ad426f8ee",
    "token": "12345"
}

async def connect_and_send():
    async with websockets.connect(WS_URL) as websocket:
        print("✅ Connected to WebSocket")

        # Send JSON (same as Postman)
        await websocket.send(json.dumps(payload))
        print("➡️ Sent:", payload)

        # Receive response
        response = await websocket.recv()
        print("⬅️ Received:", response)

asyncio.run(connect_and_send())
