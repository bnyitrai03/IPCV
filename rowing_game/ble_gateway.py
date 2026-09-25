import asyncio
import logging
import threading
from bleak import BleakScanner
from .ble_message import Message

logger = logging.getLogger("ble_gateway")

class BleGateway:
    """
    BLE Gateway that runs in its own thread and communicates with the game via a queue.
    Sends paddle state updates (left_paddle, right_paddle) to the game.
    """
    
    def __init__(self, message: Message):
        # Constants for paddle configuration
        self.PADDLE_SERVICE_UUID = "0000feaa-0000-1000-8000-00805f9b34fb"
        self.PADDLE_LEFT = "CD:35:60:84:D4:E3"
        self.PADDLE_RIGHT = "C6:43:EA:BC:7A:D4"
        
        self.message = message
        self.running = True
        self.thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.thread.start()
        logger.info("BLE Gateway started")
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("BLE Gateway stopped")
    
    def _run_async_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._listen_to_paddles())
        except Exception as e:
            logger.error(f"Error in BLE gateway async loop: {e}")
        finally:
            loop.close()
        
    # TODO: Forward the data through the message object to the game
    async def _listen_to_paddles(self):
        """Listen to paddle state changes."""
        def callback(device, advertisement_data):

            if device.address == self.PADDLE_LEFT:
                logger.info(f"Left: {advertisement_data.service_data}")
                self.message.LEFT = True

            if device.address == self.PADDLE_RIGHT:
                logger.info(f"Right: {advertisement_data.service_data}")
                self.message.RIGHT = True
                
            logging.info("-" * 40)
            
        scanner = BleakScanner(detection_callback=callback, service_uuids=[self.PADDLE_SERVICE_UUID])
        await scanner.start()
        self.message.CONFIGURED.set()
        logger.info("Started listening to paddles")
        try:
            while self.running:
                await asyncio.sleep(1)
        finally:
            await scanner.stop()