"""Stub: debug tracking removed for minimal RunPod deployment."""


class Tracker:
    def __init__(self, enabled: bool = False, maxlen: int = 100):
        self.enabled = enabled

    def reset(self):
        pass

    def track(self, *args, **kwargs):
        pass
