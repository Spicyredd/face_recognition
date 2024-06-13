import cv2
import time
import threading
from queue import Queue, Empty

class FPSThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.frame_times = []
        self.fps_queue = Queue(maxsize=10)  # limit the queue size
        self.running = True

    def run(self):
        while self.running:
            # Sleep briefly to allow frame processing
            time.sleep(0.1)
            if len(self.frame_times) > 1:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                fps = 1 / avg_frame_time if avg_frame_time > 0 else 0
                if not self.fps_queue.full():
                    self.fps_queue.put(fps)

    def stop(self):
        self.running = False

# Initialize video capture
vid = cv2.VideoCapture(1)

# Initialize FPS thread
fps_thread = FPSThread()
fps_thread.start()

prev_time = time.time()

while True:
    ret, frame = vid.read()
    if not ret:
        break

    # Calculate time difference between frames
    curr_time = time.time()
    time_diff = curr_time - prev_time
    prev_time = curr_time

    # Update frame times for FPS calculation
    fps_thread.frame_times.append(time_diff)
    if len(fps_thread.frame_times) > 10:  # Limit to the last 10 frame times
        fps_thread.frame_times.pop(0)

    # Get FPS from the queue with a timeout
    try:
        fps = fps_thread.fps_queue.get(timeout=0.1)
    except Empty:
        fps = 0

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
fps_thread.stop()
fps_thread.join()
vid.release()
cv2.destroyAllWindows()
