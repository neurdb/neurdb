import asyncio

from quart import websocket


class WebsocketSender:
    message_queue = asyncio.Queue()
    active = False

    @staticmethod
    async def start_websocket_sender_task():
        WebsocketSender.active = True
        WebsocketSender.message_queue = asyncio.Queue()
        while WebsocketSender.active:
            data = await WebsocketSender.message_queue.get()
            if data:
                await websocket.send(data)

    @staticmethod
    async def send(message):
        if not WebsocketSender.active:
            return
        asyncio.run_coroutine_threadsafe(WebsocketSender.message_queue.put(message), asyncio.get_event_loop())

    @staticmethod
    def stop():
        WebsocketSender.active = False
        WebsocketSender.message_queue = asyncio.Queue()
