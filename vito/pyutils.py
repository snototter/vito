#!/usr/bin/env python
# coding=utf-8
"""Utilities for timing, logging, etc."""

# TODO most of these generic utils should be removed from vito

import timeit
import re
import datetime
import argparse
import subprocess
import traceback
from typing import Any, Sequence


# Timing code, similar to MATLAB's tic/toc
__tictoc_timers = {}


def tic(label: str = 'default') -> None:
    """Starts a timer."""
    __tictoc_timers[label] = timeit.default_timer()


def ttoc(label: str = 'default', seconds: bool = False) -> float:
    """Stops the timer and returns the elapsed time.
    
    By default, the time will be measured in milliseconds (change to seconds
    by setting `seconds` to `True`).
    """
    if label in __tictoc_timers:
        elapsed = timeit.default_timer() - __tictoc_timers[label]
        if seconds:
            return elapsed
        else:
            return 1000.0*elapsed
    else:
        raise KeyError(f"Timer '{label}' does not exist!")


def toc(label: str = 'default', seconds: bool = False) -> None:
    """Stops the timer and prints the elapsed time.
    
    By default, the time will be measured in milliseconds (change to seconds
    by setting `seconds` to `True`).
    """
    elapsed = ttoc(label, seconds)
    print('[{:s}] Elapsed time: {:.3f} {:s}'.format(
        label, elapsed, 's' if seconds else 'ms'))


def toc_nsec(
        label: str = 'default',
        nsec: float = 0.5,
        seconds: bool = False
    ) -> None:
    """Prints the elapsed time (but mutes the output for `nsec` seconds).

    By default, the time will be measured in milliseconds (change to seconds
    by setting `seconds` to `True`).
    """
    elapsed = ttoc(label, seconds)
    s = '[{:s}] Elapsed time: {:.3f} {:s}'.format(
        label, elapsed, 's' if seconds else 'ms')
    log_nsec(s, nsec, label)


# Keep track of log timestamps to implement "log only once every x sec"
__log_timers = {}


def log_nsec(
        message: str,
        nsec: float,
        label: str = 'default'
    ) -> None:
    """Prints the message only if you haven't called `log_nsec` within the
    previous `nsec` seconds.
    
    Useful to avoid spamming your terminal.

    Args:
        message: The message string to print.
        nsec: Number of seconds (or fraction of a second) to ignore printing
            between subsequent `log_nsec` calls.
        label: Label for the used timer. Should be specified, if you want
            different "mute intervals" for different components in your code.
    """
    if label in __log_timers:
        elapsed = timeit.default_timer() - __log_timers[label]
        if elapsed < nsec:
            return
    print(message)
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


def compare(a: Any, b: Any) -> int:
    """Python3 replacement for Python 2.x cmp(), see
    https://docs.python.org/3.0/whatsnew/3.0.html#ordering-comparisons
    """
    return (a > b) - (a < b)


def compare_version_strings(v1: str, v2: str) -> int:
    """Compares version strings, returns -1/0/+1 if v1 less/equal/greater v2"""
    # Based on https://stackoverflow.com/a/1714190/400948
    def normalize_version_string(v):
        return [int(x) for x in re.sub(r'(\.0+)*$', '', v).split(".")]
    return compare(normalize_version_string(v1), normalize_version_string(v2))


# Make unicode strings, works for Python 2 & 3
try:
    to_unicode = unicode
except NameError:
    to_unicode = str


def slugify(s: str) -> str:
    """Converts a string to a slug.
    
    Strips special characters, replace white space by hyphens, converts the
    string to lowercase, etc.
    Useful to get slugs for file names or URLs.
    """
    import unicodedata
    s = unicodedata.normalize('NFKD', to_unicode(s)).encode('ascii', 'ignore').decode('ascii')
    s = to_unicode(re.sub(r'[^\w\s-]', '', s).strip().lower())
    s = to_unicode(re.sub(r'[-\s]+', '-', s))
    return s


def find_first_index(l: list, x: Any) -> int:
    """Returns the first index of element x within the list l.
    
    Raises:
        ValueError if x is not contained in the list
    """
    for idx in range(len(l)):
        if l[idx] == x:
            return idx
    raise ValueError("'{}' is not in list".format(x))


def find_last_index(l: list, x: Any) -> int:
    """Returns the last index of element x within the list l.
    
    Raises:
        ValueError if x is not contained in the list
    """
    for idx in reversed(range(len(l))):
        if l[idx] == x:
            return idx
    raise ValueError("'{}' is not in list".format(x))


def argsort(
        seq: Sequence[Any],
        indices_only: bool = False
    ) -> Any:
    """Returns the sorted indices and optionally the sorted sequence (If
    `indices_only=False`).
    """
    if indices_only:
        return sorted(range(len(seq)), key=seq.__getitem__)
    else:
        from operator import itemgetter
        return zip(*sorted(enumerate(seq), key=itemgetter(1)))


def is_tool(name: str) -> bool:
    """Check whether 'name' is on PATH and marked as executable."""
    from shutil import which
    return which(name) is not None


def safe_shell_output(*args):
    """Executes the given shell command and returns the output
    with leading/trailing whitespace trimmed.
    
    For example:
    * sso('ls')
    * sso('ls', '-l', '-a')

    Returns the tuple (True/False, output/error_message)
    """
    try:
        by = subprocess.check_output(list(args))
        out = by.decode('utf-8').strip()
        success = True
    except:
        out = traceback.format_exc(limit=3)
        success = False
    return success, out


def date_str(
        delimiter: list = ['', '', '-', '', ''],
        dt: datetime.datetime = None
    ) -> str:
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

def check_positive_int(value: Any) -> Any:
    iv = int(value)
    if iv <= 0:
        raise argparse.ArgumentTypeError("%s must be > 0" % value)
    return iv


def check_positive_real(value: Any) -> Any:
    fv = float(value)
    if fv <= 0:
        raise argparse.ArgumentTypeError("%s must be > 0.0" % value)
    return fv
