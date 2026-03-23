from stratacache.backend.base import MemoryLayer

class DiskMemoryLayer(MemoryLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

DiskStore = DiskMemoryLayer