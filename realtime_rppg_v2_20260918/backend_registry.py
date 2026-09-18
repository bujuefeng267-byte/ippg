"""One globally selected live policy, never selected using video identity/HR."""
import os

DEFAULT_MODE = 'fast'
MODES = ('fast','motion','guarded')

def selected_mode():
    mode=os.environ.get('RPPG_REALTIME_MODE',DEFAULT_MODE)
    if mode not in MODES:
        raise ValueError('RPPG_REALTIME_MODE must be fast, motion or guarded')
    return mode

def startup_seconds():
    return {'fast':10,'motion':14,'guarded':12}[selected_mode()]

def mode_description():
    return {'fast':'快速模式：窗口结束后计算。',
            'motion':'运动组合实验：窗口末端至少延后 2 秒，可能减少输出。',
            'guarded':'保守融合实验：窗口末端至少延后 2 秒，部分片段误差仍可能增加。'}[selected_mode()]

def create_backend(project_root=None,config=None):
    if selected_mode()=='guarded':
        from hybrid_backend import StreamingBackend
    elif selected_mode()=='motion':
        from retained_fast_backend import StreamingBackend
    else:
        from fast_backend import StreamingBackend
    return StreamingBackend(project_root,config)
