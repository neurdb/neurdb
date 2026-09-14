import asyncio
import json

from neurdbrt.log import logger
from quart import websocket


class WebsocketSender:
    """Per-connection outbound message pump.

    One instance is created per websocket connection (``handle_ws``) and
    registered under that connection's session id, so concurrent clients each
    get their own queue + sender loop. The previous class-level singleton
    design broke under concurrency: a second connection replaced the shared
    queue (cross-client message routing) and any client's disconnect stopped
    the sender for everyone, hanging the remaining tasks.

    Every outbound message embeds its ``sessionId``, which the static
    :meth:`send` uses to route the message to the right connection.
    """

    _by_session = {}

    def __init__(self):
        self._queue = asyncio.Queue()
        self._active = True

    def register(self, session_id: str):
        WebsocketSender._by_session[session_id] = self

    async def run(self):
        """Drain the queue into this connection's websocket.

        Started with ``asyncio.create_task`` inside the connection handler, so
        the task inherits the websocket context of its own connection.
        """
        while self._active:
            data = await self._queue.get()
            if data is None:
                break
            logger.debug(f"Sending: {data}")
            await websocket.send(data)

    def stop(self, session_id=None):
        self._active = False
        self._queue.put_nowait(None)
        if session_id is not None:
            WebsocketSender._by_session.pop(session_id, None)

    @staticmethod
    async def send(message):
        try:
            session_id = json.loads(message).get("sessionId")
        except (TypeError, ValueError):
            session_id = None
        sender = WebsocketSender._by_session.get(session_id)
        if sender is None or not sender._active:
            logger.warning(
                f"no active websocket sender for session {session_id}; message dropped"
            )
            return
        await sender._queue.put(message)
