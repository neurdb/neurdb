import asyncio

from quart import websocket


class WebsocketSender:
    message_queue = asyncio.Queue()

    @staticmethod
    async def websocket_sender_task():
        while True:
            data = await WebsocketSender.message_queue.get()
            if data:
                await websocket.send(data)

    @staticmethod
    async def send_message(message):
        asyncio.run_coroutine_threadsafe(WebsocketSender.message_queue.put(message), asyncio.get_event_loop())
