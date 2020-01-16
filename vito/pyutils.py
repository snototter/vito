#!/usr/bin/env python
# coding=utf-8
"""Utilities for timing, logging, etc."""

import timeit
import re
import sys
import os
import datetime
import argparse
import subprocess
import traceback


# Timing code, similar to MATLAB's tic/toc
__tictoc_timers = {}


def tic(label='default'):
    """Start a timer."""
    __tictoc_timers[label] = timeit.default_timer()


def toc(label='default', seconds=False):
    """Stop timer and print elapsed time."""
    if label in __tictoc_timers:
        elapsed = timeit.default_timer() - __tictoc_timers[label]
        if seconds:
            print('[{:s}] Elapsed time: {:.3f} s'.format(label, elapsed))
        else:
            print('[{:s}] Elapsed time: {:.2f} ms'.format(label, 1000.0*elapsed))


def ttoc(label='default', seconds=False):
    """Stop timer and return elapsed time."""
    if label in __tictoc_timers:
        elapsed = timeit.default_timer() - __tictoc_timers[label]
        if seconds:
            return elapsed
        else:
            return 1000.0*elapsed


def toc_nsec(label='default', nsec=0.5, seconds=False):
    """Stop timer and print elapsed time (mute output for nsec seconds)."""
    if label in __tictoc_timers:
        elapsed = timeit.default_timer() - __tictoc_timers[label]
        if seconds:
            s = '[{:s}] Elapsed time: {:.3f} s'.format(label, elapsed)
        else:
            s = '[{:s}] Elapsed time: {:.2f} ms'.format(label, 1000.0*elapsed)
        log_nsec(s, nsec, label)


# Log only once every x sec
__log_timers = {}


def log_nsec(string, nsec, label='default'):
    """Display 'string' only once every nsec seconds (floating point number). Use it to avoid spamming your terminal."""
    if label in __log_timers:
        elapsed = timeit.default_timer() - __log_timers[label]
        if elapsed < nsec:
            return
    print(string)
    __log_timers[label] = timeit.default_timer()


# Math
# def rand_mod(m):
#    """Correctly sample a random number modulo m (avoiding modulo bias)"""
#    # python's random lib has random.uniform(a,b), a <= N <= b
#    return random.uniform(0, m-1)
#
# Problem in C/C++:
#   rand() returns a number in [0, RAND_MAX], assume RAND_MAX=10, we want mod 3:
#   rand() = 0, 3, 6, 9;  then mod3 = 0; prob(0) = 4/11
#   rand() = 1, 4, 7, 10; then mod3 = 1; prob(1) = 4/11
#   rand() = 2, 5, 8; then mod3 = 2; prob(2) = 3/11 !!!
#  see also: https://stackoverflow.com/a/10984975/400948


def compare(a, b):
    """Python3 replacement for Python 2.x cmp(), see
    https://docs.python.org/3.0/whatsnew/3.0.html#ordering-comparisons
    """
    return (a > b) - (a < b)


def compare_version_strings(v1, v2):
    """Compares version strings, returns -1/0/+1 if v1 less, equal or greater v2"""
    # Based on https://stackoverflow.com/a/1714190/400948
    def normalize_version_string(v):
        return [int(x) for x in re.sub(r'(\.0+)*$', '', v).split(".")]
    return compare(normalize_version_string(v1), normalize_version_string(v2))


# Make unicode strings, works for Python 2 & 3
try:
    to_unicode = unicode
except NameError:
    to_unicode = str


def slugify(s):
    """Converts a string to a slug (strip special characters,
    replace white space, convert to lowercase...) to be used for file names or
    URLs."""
    import unicodedata
    s = unicodedata.normalize('NFKD', to_unicode(s)).encode('ascii', 'ignore').decode('ascii')
    s = to_unicode(re.sub(r'[^\w\s-]', '', s).strip().lower())
    s = to_unicode(re.sub(r'[-\s]+', '-', s))
    return s


def find_first_index(l, x):
    """Returns the first index of element x within the list l."""
    for idx in range(len(l)):
        if l[idx] == x:
            return idx
    raise ValueError("'{}' is not in list".format(x))


def find_last_index(l, x):
    """Returns the last index of element x within the list l"""
    for idx in reversed(range(len(l))):
        if l[idx] == x:
            return idx
    raise ValueError("'{}' is not in list".format(x))


def argsort(seq, indices_only=False):
    """Returns the sorted indices and the sorted array (seq) if indices_only=False."""
    if indices_only:
        return sorted(range(len(seq)), key=seq.__getitem__)
    else:
        from operator import itemgetter
        return zip(*sorted(enumerate(seq), key=itemgetter(1)))


def in_ospath(name):
    """Check whether 'name' is on PATH."""
    # Search the PATH variable, taken from https://stackoverflow.com/a/5227009
    for path in os.environ['PATH'].split(os.pathsep):
        if os.path.exists(os.path.join(path, name)):
            return True
    return False


def is_tool(name):
    """Check whether 'name' is on PATH and marked as executable."""
    if sys.version_info >= (3, 3):
        # Taken from https://stackoverflow.com/a/34177358
        from shutil import which
        return which(name) is not None
    else:
        return in_ospath(name)  # pragma: no cover


def safe_shell_output(*args):
    """Executes the given shell command and returns the output
    with leading/trailing whitespace trimmed. For example:
    * sso('ls')
    * sso('ls', '-l', '-a')
    Returns the tuple (True/False, output/error_message)
    """
    try:
        # with open(os.devnull, 'wb') as devnull:
        #     by = subprocess.check_output(list(args), stderr=devnull)
        by = subprocess.check_output(list(args))
        out = by.decode('utf-8').strip()
        success = True
    except:
        out = traceback.format_exc(limit=3)
        success = False
    return success, out


def date_str(delimiter=['', '', '-', '', ''], dt=None):
    """Returns a YYYY*MM*DD*hh*mm*ss string using the given delimiters.
    Provide less delimiter to return shorter strings, e.g.
    delimiter=['-'] returns YYYY-MM
    delimiter=['',''] returns YYYYMMDD
    etc.

    Useful to generate timestamped output folder/file names.
    You can pass a custom datetime.datetime object dt. If dt is None,
    datetime.datetime.now() will be taken
    """
    if dt is None:
        now = datetime.datetime.now()
    else:
        now = dt
    res_str = now.strftime('%Y')
    month = now.strftime('%m')
    day = now.strftime('%d')
    hour = now.strftime('%H')
    minute = now.strftime('%M')
    sec = now.strftime('%S')
    num_delim = len(delimiter)
    if num_delim == 0:
        return res_str
    res_str += '{:s}{:s}'.format(delimiter[0], month)
    if num_delim == 1:
        return res_str
    res_str += '{:s}{:s}'.format(delimiter[1], day)
    if num_delim == 2:
        return res_str
    res_str += '{:s}{:s}'.format(delimiter[2], hour)
    if num_delim == 3:
        return res_str
    res_str += '{:s}{:s}'.format(delimiter[3], minute)
    if num_delim == 4:
        return res_str
    res_str += '{:s}{:s}'.format(delimiter[4], sec)
    if num_delim > 5:
        raise RuntimeError('Too many delimiter, currently we only support formating up until seconds')
    return res_str


################################################################################
# Data validation (e.g. argument parsing)

def check_positive_int(value):
    iv = int(value)
    if iv <= 0:
        raise argparse.ArgumentTypeError("%s must be > 0" % value)
    return iv


def check_positive_real(value):
    fv = float(value)
    if fv <= 0:
        raise argparse.ArgumentTypeError("%s must be > 0.0" % value)
    return fv
