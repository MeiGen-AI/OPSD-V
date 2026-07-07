__all__ = [
    "OPSDStreamingTrainer",
]


def __getattr__(name):
    if name == "OPSDStreamingTrainer":
        from .opsd_streaming import Trainer as OPSDStreamingTrainer

        return OPSDStreamingTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
