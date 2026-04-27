import asyncio
import json
import websockets

WS_URL = "wss://webmessaging.mypurecloud.de/v1?deploymentId="  

payload = {
    "action": "getConfiguration",
    "deploymentId": ""
}

async def connect_and_send():
    async with websockets.connect(WS_URL) as websocket:
        print("Connected to WebSocket")

        # Send JSON (same as Postman)
        await websocket.send(json.dumps(payload))
        print("(->)Sent:", payload)

        # Receive response
        response = await websocket.recv()
        print("(<-)Received:", response)

asyncio.run(connect_and_send())
