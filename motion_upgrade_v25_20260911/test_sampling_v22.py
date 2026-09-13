"""Synthetic-only tests of fixed 10% trimmed RGB aggregation.

The sub-LSB pulse is a known injected array signal, not a human PPG reference.
No video, physiological label, detector threshold or interpolation is changed.
"""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import motion_frontend as frontend
import test_frontend_v2 as fixtures


def bounds(name):
    a,b,c,d=frontend.REGION_BOXES[name]
    return int(a*200),int(b*200),int(c*200),int(d*200)


def region_pixels(image, name):
    a,b,c,d=bounds(name)
    return image[b:d,a:c].reshape(-1,3).copy()


def paint_pixels(image, name, pixels):
    a,b,c,d=bounds(name)
    image[b:d,a:c]=pixels.reshape(d-b,c-a,3)


def gradient_frame(green_pulse=0.):
    """Spatially distributed quantization phases; median stays in one bin."""
    image=np.zeros((200,200,3),np.uint8)
    for name in frontend.REGION_NAMES:
        a,b,c,d=bounds(name); count=(c-a)*(d-b)
        green=np.linspace(80.25,140.25,count,dtype=np.float64)
        pixels=np.stack([green+10,green+green_pulse,green-10],axis=1)
        paint_pixels(image,name,np.rint(pixels).astype(np.uint8))
    return image


class TrimmedSamplingTests(unittest.TestCase):
    def setUp(self):
        self.landmarks=fixtures.landmarks_for_box((0,0,200,200))

    def test_sub_lsb_pulses_survive_quantization_without_reference_tuning(self):
        t=np.arange(120)/30.
        for bpm in (84.,138.):
            with self.subTest(bpm=bpm):
                injected=.15*np.sin(2*np.pi*bpm/60*t)
                trimmed=[]; median=[]
                for pulse in injected:
                    frame=gradient_frame(pulse)
                    trimmed.append(frontend.sample_rois(frame,self.landmarks)['forehead_g'])
                    median.append(frontend.sample_rois(frame,self.landmarks,rgb_aggregation='median')['forehead_g'])
                trimmed=np.asarray(trimmed); median=np.asarray(median)
                self.assertEqual(np.ptp(median),0.)
                self.assertGreater(np.ptp(trimmed),.28)
                self.assertGreater(np.corrcoef(trimmed,injected)[0,1],.99)
                self.assertLess(np.max(abs((trimmed-trimmed.mean())-(injected-injected.mean()))),.006)
                self.assertTrue(np.any(np.abs(trimmed*2-np.rint(trimmed*2))>1e-6))

    def test_less_than_ten_percent_valid_highlights_do_not_bias_constant_patch(self):
        image,landmarks,colors=fixtures.painted_frame()
        for name in frontend.REGION_NAMES:
            pixels=region_pixels(image,name)
            count=int(.09*len(pixels))
            pixels[:count]=[240,240,240]  # Valid under the unchanged 20..245 mask.
            paint_pixels(image,name,pixels)
            self.assertGreater(pixels.mean(axis=0)[0],colors[name][0]+1)
        sampled=frontend.sample_rois(image,landmarks)
        for name,color in colors.items():
            np.testing.assert_array_equal([sampled[f'{name}_{c}'] for c in 'rgb'],color)
            self.assertEqual(sampled[f'{name}_quality'],1.)
            self.assertEqual(sampled[f'{name}_rgb_aggregation'],'trimmed_mean')

    def test_finite_mask_and_24_25_pixel_boundary_are_unchanged(self):
        image=np.zeros((200,200,3),np.float64)
        pixels=region_pixels(image,'forehead')
        legal=np.column_stack([21+np.arange(25),80+np.arange(25),130+np.arange(25)]).astype(float)
        pixels[:24]=legal[:24]
        pixels[25:30]=[[20,100,100],[245,100,100],[100,np.nan,100],[100,100,np.inf],[0,100,100]]
        paint_pixels(image,'forehead',pixels)
        before=frontend.sample_rois(image,self.landmarks)
        self.assertEqual(before['forehead_valid_pixels'],24)
        self.assertFalse(before['forehead_valid'])
        pixels[24]=legal[24];paint_pixels(image,'forehead',pixels)
        after=frontend.sample_rois(image,self.landmarks)
        self.assertEqual(after['forehead_valid_pixels'],25)
        self.assertTrue(after['forehead_valid'])
        np.testing.assert_array_equal([after[f'forehead_{c}'] for c in 'rgb'],legal[2:-2].mean(axis=0))
        self.assertAlmostEqual(after['forehead_quality'],25/len(pixels))

    def test_constant_patch_and_invalid_frames_do_not_invent_pulses(self):
        for mode in frontend.RGB_AGGREGATIONS:
            with self.subTest(mode=mode):
                for level in (40.,100.125,200.):
                    sample=frontend.sample_rois(np.full((200,200,3),level),self.landmarks,mode)
                    np.testing.assert_allclose([sample[c] for c in 'rgb'],level,rtol=0,atol=1e-12)
                for level in (0.,255.,np.nan,np.inf):
                    sample=frontend.sample_rois(np.full((200,200,3),level),self.landmarks,mode)
                    self.assertFalse(sample['rgb_valid'])
                    self.assertTrue(np.isnan(sample['g']))
                    self.assertTrue(all(sample[f'{r}_quality']==0 for r in frontend.REGION_NAMES))

    def test_float64_precision_and_integer_input_conversion(self):
        uint8=gradient_frame(.1)
        integer=frontend.sample_rois(uint8,self.landmarks)
        floating=frontend.sample_rois(uint8.astype(np.float64),self.landmarks)
        for key in ('r','g','b',*[f'{r}_{c}' for r in frontend.REGION_NAMES for c in 'rgb']):
            self.assertEqual(integer[key],floating[key])
        image=np.full((200,200,3),100.000000125,np.float64)
        sample=frontend.sample_rois(image,self.landmarks)
        self.assertAlmostEqual(sample['g'],100.000000125,places=11)
        self.assertNotEqual(sample['g'],100.)

    def test_sampling_flags_geometry_and_quality_do_not_depend_on_aggregation(self):
        image=gradient_frame(.12)
        trim=frontend.sample_rois(image,self.landmarks,'trimmed_mean')
        median=frontend.sample_rois(image,self.landmarks,'median')
        for name in frontend.REGION_NAMES:
            for key in ('valid','quality','valid_pixel_fraction','visible_fraction','pixel_count',
                        'valid_pixels','brightness','laplacian_var','status'):
                self.assertEqual(trim[f'{name}_{key}'],median[f'{name}_{key}'])
        self.assertEqual(trim['roi_valid_regions'],median['roi_valid_regions'])

    def test_median_ablation_matches_previous_rgb_sampler(self):
        for pulse in (0.,.15,-.15):
            image=gradient_frame(pulse)
            sampled=frontend.sample_rois(image,self.landmarks,'median')
            legacy=frontend.legacy.roi_mean_rgb(image,self.landmarks)
            np.testing.assert_array_equal([sampled[c] for c in 'rgb'],legacy)

    def test_invalid_aggregation_fails_before_video_access(self):
        with self.assertRaises(ValueError):
            frontend.sample_rois(np.zeros((20,20,3)),None,'adaptive_to_reference')
        with patch.object(frontend.cv2,'VideoCapture') as capture:
            with self.assertRaises(ValueError):
                frontend.extract('unused_video',rgb_aggregation='mean')
            capture.assert_not_called()

    def test_extract_forwards_aggregation_and_records_it(self):
        image=gradient_frame(.1)
        for mode in frontend.RGB_AGGREGATIONS:
            with self.subTest(mode=mode):
                class Capture:
                    count=0
                    def isOpened(self):return True
                    def get(self,key):return 30.
                    def read(self):
                        self.count+=1
                        return (True,image[:,:,::-1].copy()) if self.count==1 else (False,None)
                    def release(self):pass
                landmarks=self.landmarks
                class Mesh:
                    def __init__(self,**kwargs):pass
                    def process(self,frame):return SimpleNamespace(multi_face_landmarks=[landmarks])
                    def close(self):pass
                expected=frontend.sample_rois(image,self.landmarks,mode)
                with patch.object(frontend.cv2,'VideoCapture',return_value=Capture()), \
                     patch.object(frontend.legacy.mp.solutions.face_mesh,'FaceMesh',Mesh), \
                     patch.object(frontend.legacy,'flow_affine',return_value=None):
                    trace,fps=frontend.extract('controlled_array',rgb_aggregation=mode)
                self.assertEqual(len(trace),1)
                self.assertEqual(fps,30.)
                self.assertEqual(trace.attrs['rgb_aggregation'],mode)
                for name in frontend.REGION_NAMES:
                    self.assertEqual(trace.loc[0,f'{name}_rgb_aggregation'],mode)
                    self.assertEqual(trace.loc[0,f'{name}_g'],expected[f'{name}_g'])


if __name__=='__main__':unittest.main()
