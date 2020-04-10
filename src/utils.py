import time

def print_time(method):
    
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('Time of \'{}\' : {:.5f} s'.format(method.__name__, te - ts))
        return result
    
    return timed
