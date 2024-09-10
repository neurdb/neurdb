import asyncio

from quart import websocket


class WebsocketSender:

    # Shared message queue for handling messages to be sent
    message_queue = asyncio.Queue()
    # A flag to indicate whether the WebSocket sender is active
    active = False

    @staticmethod
    async def start_websocket_sender_task():
        """
        Starts a task that continuously listens for messages from the queue
        and sends them over the WebSocket when they arrive. This loop runs
        until the 'active' flag is set to False.
        """
        WebsocketSender.active = True
        WebsocketSender.message_queue = asyncio.Queue()

        while WebsocketSender.active:
            data = await WebsocketSender.message_queue.get()
            if data:
                await websocket.send(data)

    @staticmethod
    async def send(message):
        """
        Adds a message to the queue to be sent via WebSocket.
        """
        if not WebsocketSender.active:
            return

        asyncio.run_coroutine_threadsafe(
            WebsocketSender.message_queue.put(message), asyncio.get_event_loop()
        )

    @staticmethod
    def stop():
        """
        Exit task loop to exit & resets the message queue.
        """
        WebsocketSender.active = False  # Set the sender as inactive
        WebsocketSender.message_queue = asyncio.Queue()  # Reset the queue
