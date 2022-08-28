from timeit import default_timer as timer


def decorator_timer(function):
    def wrapper(*args, **kwargs):
        start = timer()
        result = function(*args, **kwargs)
        elapsed = (timer() - start)
        
        return result, elapsed
    return wrapper
