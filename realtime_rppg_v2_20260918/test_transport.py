import hashlib
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request,urlopen

import cv2
import numpy as np

from app import decode_frame, raw_frame_size, RAW_CONTENT_TYPE, handler_for


class FakeSession:
    id='owned-session'
    def ingest(self,frame,timestamp,**kwargs):
        self.frame=frame.copy();self.timestamp=timestamp
        return {'ok':True,'shape':list(frame.shape),'timestamp':timestamp,
                'frame_bytes':kwargs['byte_count'],**kwargs}


class FakeService:
    token='local-token'
    current=FakeSession()


class TransportTests(unittest.TestCase):
    def test_persistent_frames_share_connection_and_unread_rejection_closes(self):
        service=FakeService()
        server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(service))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        connection=HTTPConnection('127.0.0.1',server.server_port,timeout=3)
        try:
            headers={'X-RPPG-Token':service.token,'X-Session':service.current.id,
                     'Content-Type':RAW_CONTENT_TYPE,'X-Frame-Width':'1','X-Frame-Height':'1',
                     'X-Capture-Time':'0'}
            ports=[]
            for i in range(2):
                headers['X-Capture-Time']=str(i/30)
                connection.request('POST','/api/frame',b'\x01\x02\x03\xff',headers)
                ports.append(connection.sock.getsockname()[1])
                reply=connection.getresponse()
                self.assertEqual(reply.version,11)
                self.assertEqual(reply.status,200)
                self.assertAlmostEqual(json.loads(reply.read())['timestamp'],i/30)
            self.assertEqual(ports[0],ports[1])
            bad=dict(headers,**{'X-RPPG-Token':'wrong'})
            connection.request('POST','/api/frame',b'unread request body',bad)
            reply=connection.getresponse()
            self.assertEqual(reply.status,403)
            self.assertEqual(reply.getheader('Connection'),'close')
            reply.read()
            self.assertIsNone(connection.sock)
            connection.request('POST','/api/frame',b'\x01\x02\x03\xff',headers)
            reply=connection.getresponse()
            self.assertEqual(reply.status,200)
            reply.read()
        finally:
            connection.close();server.shutdown();server.server_close();thread.join(timeout=3)

    def test_raw_preserves_every_color_byte(self):
        rgba=np.random.default_rng(7).integers(0,256,(54,96,4),dtype=np.uint8)
        decoded,kind=decode_frame(rgba.tobytes(),RAW_CONTENT_TYPE,'96','54')
        np.testing.assert_array_equal(decoded[:,:,::-1],rgba[:,:,:3])
        self.assertEqual(kind,'rgba8')

    def test_ordered_pixel_and_channel_layout(self):
        rgba=np.array([[[1,2,3,255],[4,5,6,255]],[[7,8,9,255],[10,11,12,255]]],np.uint8)
        decoded,_=decode_frame(rgba.tobytes(),RAW_CONTENT_TYPE,'2','2')
        np.testing.assert_array_equal(decoded,np.array([[[3,2,1],[6,5,4]],[[9,8,7],[12,11,10]]],np.uint8))

    def test_strict_dimensions(self):
        for w,h in [(None,'3'),('3.5','3'),('-1','3'),('0','3'),('4097','3'),('3000','3000'),('３','3')]:
            with self.assertRaises(ValueError):raw_frame_size(w,h)

    def test_exact_byte_count(self):
        for body in (b'',b'123',b'12345'):
            with self.assertRaises(ValueError):decode_frame(body,RAW_CONTENT_TYPE,'1','1')

    def test_jpeg_compatibility_and_unsupported_content(self):
        src=np.full((54,96,3),100,np.uint8)
        ok,body=cv2.imencode('.jpg',src,[cv2.IMWRITE_JPEG_QUALITY,98]);self.assertTrue(ok)
        decoded,kind=decode_frame(body.tobytes(),'image/jpeg')
        self.assertEqual(kind,'jpeg');self.assertEqual(decoded.shape,src.shape)
        for content in ('image/png','application/octet-stream'):
            with self.assertRaises(ValueError):decode_frame(body.tobytes(),content)
        with self.assertRaises(ValueError):decode_frame(b'not jpeg','image/jpeg')

    def test_http_raw_transport_and_auth(self):
        service=FakeService()
        server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(service))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            rgba=np.full((20,30,4),127,np.uint8)
            url=f'http://127.0.0.1:{server.server_port}/api/frame'
            headers={'X-RPPG-Token':service.token,'X-Session':service.current.id,
                     'Content-Type':RAW_CONTENT_TYPE,'X-Frame-Width':'30','X-Frame-Height':'20',
                     'X-Capture-Time':'1.25'}
            with urlopen(Request(url,rgba.tobytes(),headers=headers),timeout=3) as r: result=json.load(r)
            self.assertEqual(result['transport'],'rgba8')
            self.assertEqual(result['frame_bytes'],20*30*4)
            self.assertEqual(service.current.timestamp,1.25)
            np.testing.assert_array_equal(service.current.frame,rgba[:,:,:3])
            bad=dict(headers,**{'X-Frame-Width':'31'})
            with self.assertRaises(HTTPError) as failure:urlopen(Request(url,rgba.tobytes(),headers=bad),timeout=3)
            self.assertEqual(failure.exception.code,400)
            bad=dict(headers,**{'X-RPPG-Token':'wrong'})
            with self.assertRaises(HTTPError) as failure:urlopen(Request(url,rgba.tobytes(),headers=bad),timeout=3)
            self.assertEqual(failure.exception.code,403)
        finally:
            server.shutdown();server.server_close();thread.join(timeout=3)


if __name__=='__main__':unittest.main()
