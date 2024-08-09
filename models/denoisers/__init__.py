__DENOISERS__ = {}


def register_denoiser(name: str):
    def wrapper(cls):
        if __DENOISERS__.get(name, None):
            raise NameError(f"Name {name} is already registered.")
        __DENOISERS__[name] = cls
        return cls

    return wrapper


def get_denoiser(name: str):
    if __DENOISERS__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __DENOISERS__[name]


import os

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("__"):
        __import__(f"{__name__}.{file[:-3]}", locals(), globals())
