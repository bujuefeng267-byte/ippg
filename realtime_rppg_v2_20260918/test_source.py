import threading
import unittest
from run_source import LatestFrameCapture


class FakeCapture:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.released = False
        self.release_thread = None

    def set(self, key, value):
        return False

    def read(self):
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None

    def release(self):
        self.released = True
        self.release_thread = threading.current_thread().name


class TestLatestCapture(unittest.TestCase):
    def test_slow_consumer_gets_latest_and_bounded_slot(self):
        cap = FakeCapture(range(100))
        ticks = iter(range(101))
        reader = LatestFrameCapture(cap, clock=lambda: next(ticks))
        reader.thread.join(1)
        stats = reader.snapshot()
        self.assertEqual(stats['capture_read_frames'], 100)
        self.assertEqual(stats['capture_overwritten_frames'], 99)
        self.assertEqual(stats['capture_pending_frames'], 1)
        self.assertEqual(reader.read(), (99, 99))
        self.assertIsNone(reader.read())
        self.assertTrue(reader.close())
        self.assertEqual(cap.release_thread, 'rppg-live-capture')

    def test_worker_error_is_reported_and_released(self):
        cap = FakeCapture([])
        def broken_read():
            raise RuntimeError('capture failed')
        cap.read = broken_read
        reader = LatestFrameCapture(cap)
        reader.thread.join(1)
        with self.assertRaisesRegex(RuntimeError, 'capture failed'):
            reader.read()
        self.assertTrue(cap.released)
        self.assertTrue(reader.close())

    def test_blocked_read_is_not_released_concurrently(self):
        cap = FakeCapture([])
        entered, unblock = threading.Event(), threading.Event()
        def blocked_read():
            entered.set()
            unblock.wait()
            return False, None
        cap.read = blocked_read
        reader = LatestFrameCapture(cap)
        self.assertTrue(entered.wait(1))
        self.assertFalse(reader.close(timeout_s=.01))
        self.assertFalse(cap.released)
        unblock.set()
        self.assertTrue(reader.close(timeout_s=1))
        self.assertTrue(cap.released)


if __name__ == '__main__':
    unittest.main()
