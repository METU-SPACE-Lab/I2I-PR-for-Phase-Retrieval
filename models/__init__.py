__MODELS__ = {}


def register_model(name: str):
    def wrapper(cls):
        if __MODELS__.get(name, None):
            raise NameError(f"Name {name} is already registered.")
        __MODELS__[name] = cls
        return cls

    return wrapper


def get_model(name: str):
    if __MODELS__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __MODELS__[name]
