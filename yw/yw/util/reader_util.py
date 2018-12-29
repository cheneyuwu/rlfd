import pandas

# ================================================================
# Readers
# ================================================================

def read_json(fname):
    ds = []
    with open(fname, 'rt') as fh:
        for line in fh:
            ds.append(json.loads(line))
    return pandas.DataFrame(ds)

def read_csv(fname):
    return pandas.read_csv(fname, index_col=None, comment='#')