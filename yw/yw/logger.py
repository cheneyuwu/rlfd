import os
import sys
import shutil
import os.path as osp
import json
import time
import datetime
import tempfile
from collections import defaultdict

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50

class KVWriter(object):
    def writekvs(self, kvs):
        raise NotImplementedError

class SeqWriter(object):
    def writeseq(self, seq):
        raise NotImplementedError

class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, 'wt')
            self.own_file = True
        else:
            assert hasattr(filename_or_file, 'read'), 'expected file or str, got %s'%filename_or_file
            self.file = filename_or_file
            self.own_file = False

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = {}
        for (key, val) in sorted(kvs.items()):
            if isinstance(val, float):
                valstr = '%-8.3g' % (val,)
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        if len(key2str) == 0:
            print('WARNING: tried to write empty key-value dict')
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = '-' * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
            lines.append('| %s%s | %s%s |' % (
                key,
                ' ' * (keywidth - len(key)),
                val,
                ' ' * (valwidth - len(val)),
            ))
        lines.append(dashes)
        self.file.write('\n'.join(lines) + '\n')

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, s):
        return s[:20] + '...' if len(s) > 23 else s

    def writeseq(self, seq):
        seq = list(seq)
        for (i, elem) in enumerate(seq):
            self.file.write(elem)
            if i < len(seq) - 1: # add space unless this is the last one
                self.file.write(' ')
        self.file.write('\n')
        self.file.flush()

    def close(self):
        if self.own_file:
            self.file.close()

def make_output_format(format, ev_dir, log_suffix=''):
    os.makedirs(ev_dir, exist_ok=True)
    if format == 'stdout':
        return HumanOutputFormat(sys.stdout)
    elif format == 'log':
        return HumanOutputFormat(osp.join(ev_dir, 'log%s.txt' % log_suffix))
    # Yuchen: add json, csv and tensorboard format later
    else:
        raise ValueError('Unknown format specified: %s' % (format,))

# ================================================================
# API
# ================================================================

def dumpkvs():
    """
    Write all of the diagnostics from the current iteration

    level: int. (see logger.py docs) If the global logger level is higher than
                the level argument here, don't print to stdout.
    """
    Logger.CURRENT.dumpkvs()

def getkvs():
    return Logger.CURRENT.name2val

def get_dir():
    """
    Get directory that log files are being written to.
    will be None if there is no output directory (i.e., if you didn't call start)
    """
    return Logger.CURRENT.get_dir()

def log(*args, level=INFO):
    """
    Write the sequence of args, with no separators, to the console and output files (if you've configured an output file).
    """
    Logger.CURRENT.log(*args, level=level)

def logkvs(d):
    """
    Log a dictionary of key-value pairs
    """
    for (k, v) in d.items():
        logkv(k, v)

def logkv(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.
    """
    Logger.CURRENT.logkv(key, val)

def logkv_mean(key, val):
    """
    The same as logkv(), but if called many times, values averaged.
    """
    Logger.CURRENT.logkv_mean(key, val)

def set_level(level):
    """
    Set logging threshold on current logger.
    """
    Logger.CURRENT.set_level(level)

def debug(*args):
    log(*args, level=DEBUG)

def info(*args):
    log(*args, level=INFO)

def warn(*args):
    log(*args, level=WARN)

def error(*args):
    log(*args, level=ERROR)

# ================================================================
# Backend
# ================================================================

class Logger(object):
    DEFAULT = None  # A logger with no output files. (See right below class definition)
                    # So that you can still log to the terminal without setting up any output files
    CURRENT = None  # Current logger being used by the free functions above

    def __init__(self, dir, output_formats):
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)
        self.level = INFO
        self.dir = dir
        self.output_formats = output_formats

    # Logging API, forwarded
    # ----------------------------------------
    def log(self, *args, level=INFO):
        if self.level <= level:
            self._do_log(args)

    def logkv(self, key, val):
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        if val is None:
            self.name2val[key] = None
            return
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval*cnt/(cnt+1) + val/(cnt+1)
        self.name2cnt[key] = cnt + 1

    def dumpkvs(self):
        if self.level == DISABLED: return
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.writekvs(self.name2val)
        self.name2val.clear()
        self.name2cnt.clear()

    # Configuration
    # ----------------------------------------
    def set_level(self, level):
        self.level = level

    def get_dir(self):
        return self.dir

    def close(self):
        for fmt in self.output_formats:
            fmt.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args):
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.writeseq(map(str, args))

def configure(dir=None, format_strs=None):
    if dir is None:
        dir = os.getenv('YW_LOGDIR')
    if dir is None:
        dir = osp.join(tempfile.gettempdir(),
                       datetime.datetime.now().strftime("yw-%Y-%m-%d-%H-%M-%S-%f"))
    assert isinstance(dir, str)
    os.makedirs(dir, exist_ok=True)

    log_suffix = ''
    rank = 0
    # check environment variables here instead of importing mpi4py
    # to avoid calling MPI_Init() when this module is imported
    for varname in ['PMI_RANK', 'OMPI_COMM_WORLD_RANK']:
        if varname in os.environ:
            rank = int(os.environ[varname])
    if rank > 0:
        log_suffix = "-rank%03i" % rank

    if format_strs is None:
        if rank == 0:
            # Yuchen use stdout log only for now?
            # format_strs = os.getenv('OPENAI_LOG_FORMAT', 'stdout,log,csv').split(',')
            format_strs = ['stdout', 'log']
        else:
            # format_strs = os.getenv('OPENAI_LOG_FORMAT_MPI', 'log').split(',')
            format_strs = ['log']
    format_strs = filter(None, format_strs)
    output_formats = [make_output_format(f, dir, log_suffix) for f in format_strs]

    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats)
    log('Logging to %s' % dir)

def reset():
    if Logger.CURRENT is not Logger.DEFAULT:
        Logger.CURRENT.close()
        Logger.CURRENT = Logger.DEFAULT
        log('Reset logger')


def _configure_default_logger():
    format_strs = None
    # keep the old default of only writing to stdout
    if 'OPENAI_LOG_FORMAT' not in os.environ:
        format_strs = ['stdout']
    configure(format_strs=format_strs)
    Logger.DEFAULT = Logger.CURRENT

# ================================================================

def _demo():
    info("hi")
    debug("shouldn't appear")
    set_level(DEBUG)
    debug("should appear")
    dir = "/tmp/testlogging"
    if os.path.exists(dir):
        shutil.rmtree(dir)
    configure(dir=dir)
    logkv("a", 3)
    logkv("b", 2.5)
    dumpkvs()
    logkv("b", -2.5)
    logkv("a", 5.5)
    dumpkvs()
    info("^^^ should see a = 5.5")
    logkv_mean("b", -22.5)
    logkv_mean("b", -44.4)
    logkv("a", 5.5)
    dumpkvs()
    info("^^^ should see b = 33.3")

    logkv("b", -2.5)
    dumpkvs()

    logkv("a", "longasslongasslongasslongasslongasslongassvalue")
    dumpkvs()

record_tabular = logkv
dump_tabular = dumpkvs

# configure the default logger on import
_configure_default_logger()

if __name__ == "__main__":
    _demo()

