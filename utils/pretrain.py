def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {
        k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape
    }
