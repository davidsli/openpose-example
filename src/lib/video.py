import queue
import threading
import cv2


class BufferlessVideoCapture(cv2.VideoCapture):
    def __init__(self, name):
        super().__init__(name)
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._reader)
        self.stop_thread = False
        self.thread.daemon = True
        self.thread.start()

    def _reader(self):
        while True:
            if self.stop_thread:
                break
            ret, frame = super().read()
            if not ret:
                break
            if not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            self.queue.put(frame)

    def read(self):
        return self.queue.get()

    def release(self):
        self.stop_thread = True
        self.thread.join()
        super().release()
