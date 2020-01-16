import argparse
import datetime
import pytest
from vito.pyutils import compare_version_strings, slugify, find_first_index, \
    find_last_index, argsort, is_tool, date_str, check_positive_int, \
    check_positive_real, safe_shell_output, in_ospath, tic, toc, ttoc, \
    toc_nsec, log_nsec


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
    x = [0, 1, 17, -5, 17, 3, 3, 2, 0, -5, 'b']
    assert find_first_index(x, 0) == 0
    assert find_first_index(x, 17) == 2
    assert find_first_index(x, 3) == 5
    with pytest.raises(ValueError):
        find_first_index(x, 1000)
    with pytest.raises(ValueError):
        find_first_index(x, 'a')

    assert find_last_index(x, 0) == 8
    assert find_last_index(x, 3) == 6
    assert find_last_index(x, -5) == 9
    assert find_last_index(x, 17) == 4
    assert find_last_index(x, 2) == find_first_index(x, 2)
    with pytest.raises(ValueError):
        find_last_index(x, -1000)
    with pytest.raises(ValueError):
        find_last_index(x, 'a')


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
    with pytest.raises(RuntimeError):
        date_str(['', '', '', ':', ':', ''])


def test_is_tool():
    assert is_tool('fooblablub') is False
    assert is_tool('dir') is True
    assert in_ospath('fooblablub') is False
    assert in_ospath('dir') is True


def test_safe_shell_output():
    res, _ = safe_shell_output('ls')
    assert res
    res, _ = safe_shell_output('an-invalidcmd')
    assert res is False


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


def test_tictoc(capsys):
    import time
    tic()
    time.sleep(0.5)
    # Time into variable
    passed_ms = ttoc(seconds=False)
    passed_s = ttoc(seconds=True)
    assert passed_ms >= 500.0
    assert passed_s >= 0.5
    # Log time to stdout
    toc(seconds=False)
    captured = capsys.readouterr()
    assert captured.out.startswith('[default] Elapsed time: ')
    assert captured.out.endswith(' ms\n')
    tic('test')
    toc('test', seconds=True)
    captured = capsys.readouterr()
    assert captured.out.startswith('[test] Elapsed time: ')
    assert captured.out.endswith(' s\n')
    # Test toc_nsec
    tic(label='nsec')
    toc_nsec(label='nsec', nsec=0.5, seconds=False)
    toc_nsec(label='nsec', nsec=0.5, seconds=False)
    toc_nsec(label='nsec', nsec=0.5, seconds=False)
    captured = capsys.readouterr()
    assert captured.out.startswith('[nsec] Elapsed time: ')
    assert captured.out.endswith(' ms\n')
    assert captured.out.count('\n') == 1
    time.sleep(0.5)
    toc_nsec(label='nsec', nsec=0.5, seconds=True)
    toc_nsec(label='nsec', nsec=0.5, seconds=False)
    captured = capsys.readouterr()
    assert captured.out.startswith('[nsec] Elapsed time: ')
    assert captured.out.endswith(' s\n')
    assert captured.out.count('\n') == 1


def test_log_nsec(capsys):
    log_nsec('foo', 5)
    log_nsec('bar', 5)
    log_nsec('bla', 5)
    log_nsec('nice', 10, label='other')
    captured = capsys.readouterr()
    assert captured.out == 'foo\nnice\n'
    log_nsec('foo', 5)
    captured = capsys.readouterr()
    assert len(captured.out) == 0
    log_nsec('override', 0)
    captured = capsys.readouterr()
    assert captured.out == 'override\n'


def test_version():
    # Ensure that tests fail upon major API change
    import vito
    assert compare_version_strings(vito.__version__, "1.0") > 0
    assert compare_version_strings(vito.__version__, "2") < 0
