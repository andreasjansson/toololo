from enum import IntEnum
import sys


class LogLevel(IntEnum):
    ERROR = 6
    WARNING = 5
    INFO = 4
    VERBOSE1 = 3
    VERBOSE2 = 2
    VERBOSE3 = 1


ERROR = LogLevel.ERROR
WARNING = LogLevel.WARNING
INFO = LogLevel.INFO
VERBOSE1 = LogLevel.VERBOSE1
VERBOSE2 = LogLevel.VERBOSE2
VERBOSE3 = LogLevel.VERBOSE3

level = INFO


def error(message: str):
    if level <= ERROR:
        _print(message)


def warning(message: str):
    if level <= WARNING:
        _print(message)


def info(message: str):
    if level <= INFO:
        _print(message)


def v(message: str):
    if level <= VERBOSE1:
        _print(message)


def vv(message: str):
    if level <= VERBOSE2:
        _print(message)


def vvv(message: str):
    if level <= VERBOSE3:
        _print(message)


def set_verbosity(verbosity: LogLevel):
    global level
    if verbosity == 0:
        level = INFO
    if verbosity == 1:
        level = VERBOSE1
    if verbosity == 2:
        level = VERBOSE2
    if verbosity == 3:
        level = VERBOSE3


def _print(message):
    print(message, file=sys.stderr)
    sys.stderr.flush()
