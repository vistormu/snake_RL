class _Formatter:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def bold(cls, message: str) -> str:
        return cls.BOLD + message + cls.END

    @classmethod
    def green(cls, message: str) -> str:
        return cls.GREEN + message + cls.END

    @classmethod
    def blue(cls, message: str) -> str:
        return cls.BLUE + message + cls.END

    @classmethod
    def yellow(cls, message: str) -> str:
        return cls.YELLOW + message + cls.END

    @classmethod
    def red(cls, message: str) -> str:
        return cls.RED + message + cls.END


class Logger(_Formatter):

    muteInfo = False
    muteDebug = False
    muteWarning = False
    muteError = False
    muteAll = False

    def __init__(self, name: str):
        self.name = name

    def info(self, *message):
        name = super().bold('[' + self.name + ']')
        name = super().green(name)
        level = super().green('[INFO]')
        message = ''.join(map(str, message))
        message = super().green(message)

        if not self.muteAll and not self.muteInfo:
            print(name + level + ' ' + message)

    def debug(self, *message):
        name = super().bold('[' + self.name + ']')
        name = super().blue(name)
        level = super().blue('[DEBUG]')
        message = ''.join(map(str, message))
        message = super().blue(message)

        if not self.muteAll and not self.muteDebug:
            print(name + level + ' ' + message)

    def warning(self, *message):
        name = super().bold('[' + self.name + ']')
        name = super().yellow(name)
        level = super().yellow('[WARNING]')
        message = ''.join(map(str, message))
        message = super().yellow(message)

        if not self.muteAll and not self.muteWarning:
            print(name + level + ' ' + message)

    def error(self, *message):
        name = super().bold('[' + self.name + ']')
        name = super().red(name)
        level = super().red('[ERROR]')
        message = ''.join(map(str, message))
        message = super().red(message)

        if not self.muteAll and not self.muteError:
            print(name + level + ' ' + message)

    def init(self):
        self.info('Logger initialized')
