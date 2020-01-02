import argparse
import datetime
import pytest
from ..pyutils import compare_version_strings, slugify, find_first_index, \
    find_last_index, argsort, is_tool, date_str, check_positive_int, \
    check_positive_real


def test_compare_version_strings():
    assert compare_version_strings('0.1', '0.2') < 0
    assert compare_version_strings('0.17', '0.2') > 0
    assert compare_version_strings('1.5', '1.5') == 0
    assert compare_version_strings('1.5.0', '1.4') > 0
    assert compare_version_strings('1.4.9', '1.6') < 0


def test_slugify():
    assert slugify('') == ''
    assert slugify('abs 17#') == 'abs-17'
    assert slugify('  a  1 ') == 'a-1'


def test_find_indices():
    x = [0, 1, 17, -5, 17, 3, 3, 2, 0, -5]
    assert find_first_index(x, 0) == 0
    assert find_first_index(x, 17) == 2
    assert find_first_index(x, 3) == 5

    assert find_last_index(x, 0) == 8
    assert find_last_index(x, 3) == 6
    assert find_last_index(x, -5) == 9
    assert find_last_index(x, 17) == 4
    assert find_last_index(x, 2) == find_first_index(x, 2)


def test_argsort():
    x = ['z', 'foo', 'bar', 'xkcd']
    expected = [2, 1, 3, 0]
    sidx = argsort(x, indices_only=True)
    assert all([sidx[i] == expected[i] for i in range(len(x))])

    sidx, sx = argsort(x, indices_only=False)
    assert all([sidx[i] == expected[i] for i in range(len(x))])
    assert all(sx[i] == x[sidx[i]] for i in range(len(x)))


def test_date_str():
    ts = datetime.datetime.now()
    year = ts.strftime('%Y')
    month = ts.strftime('%m')
    day = ts.strftime('%d')
    hour = ts.strftime('%H')
    minute = ts.strftime('%M')
    sec = ts.strftime('%S')
    assert date_str([], ts) == year
    assert date_str([''], ts) == ''.join([year, month])
    assert date_str(['_-_'], ts) == '_-_'.join([year, month])
    assert date_str(['#', '#'], ts) == '#'.join([year, month, day])
    assert date_str(['!', '!', '!'], ts) == '!'.join([year, month, day, hour])
    assert date_str(['-', '-', '-', '-'], ts) == \
        '-'.join([year, month, day, hour, minute])
    assert date_str(['_', '_', '_', '_', '_'], ts) == \
        '_'.join([year, month, day, hour, minute, sec])
    assert date_str(['', '-', '<#>', ':', ':|:_#'], ts) == \
        '{}{}-{}<#>{}:{}:|:_#{}'.format(year, month, day, hour, minute, sec)


def test_is_tool():
    # TODO check if 'dir' works on Windows/Mac test servers
    assert is_tool('fooblablub') is False
    assert is_tool('dir') is True


def test_check_positive_int():
    for v in [-17, -1000000, 0, -1, -17.42, -0.0001]:
        with pytest.raises(argparse.ArgumentTypeError):
            check_positive_int(v)
    for v in [1, 1.01, 17, 23, 555.777, 10000000.01]:
        check_positive_int(v)


def test_check_positive_real():
    for v in [-17, -1000000, 0, -1.27, -17.42, -0.0001]:
        with pytest.raises(argparse.ArgumentTypeError):
            check_positive_real(v)
    for v in [0.0000001, 1, 1.01, 17, 23, 555.777, 10000000.01]:
        check_positive_real(v)
