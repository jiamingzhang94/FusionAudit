class StreamGenerationConfig(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)


class NewGenerationMixin:
    @staticmethod
    def generate(*args, **kwargs):
        raise NotImplementedError("Streaming generation is not used in this project.")

    @staticmethod
    def sample_stream(*args, **kwargs):
        raise NotImplementedError("Streaming generation is not used in this project.")


def init_stream_support(*args, **kwargs):
    return None
