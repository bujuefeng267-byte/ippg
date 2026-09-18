"""R1 numerical behavior, with a vectorized POS projection only."""
from frozen_r1_backend import StreamingBackend as BaseBackend, StreamingConfig
from fast_kernels import pos_projection

class StreamingBackend(BaseBackend):
    def __init__(self, project_root=None, config=None):
        super().__init__(project_root, config)
        self.pos_signal=lambda rgb,fps:pos_projection(rgb,fps,False)

    @property
    def configuration(self):
        out=super().configuration
        out['implementation']='vectorized_numerically_equivalent_pos'
        return out
