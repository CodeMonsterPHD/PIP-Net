import logging
import datetime
logger = logging.getLogger(__name__)


class AttrDict(dict):
    """
    Subclass dict and define getter-setter.
    This behaves as both dict and obj.
    """

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value


__C = AttrDict()
config = __C

__C.BASIC = AttrDict()
__C.BASIC.TIME = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
__C.BASIC.NUM_WORKERS = 10
__C.BASIC.SEED = ''
__C.BASIC.DISP_FREQ = 10  # frequency to display
__C.BASIC.SAVE_DIR = ''
__C.BASIC.ROOT_DIR = ''
__C.BASIC.CKPT_DIR = ''
__C.BASIC.LOG_DIR = ''
__C.BASIC.LOG_FILE = ''
__C.BASIC.BACKUP_CODES = True
__C.BASIC.BACKUP_LIST = ['lib', 'tools']

# Cudnn related setting
__C.CUDNN = AttrDict()
__C.CUDNN.BENCHMARK = False
__C.CUDNN.DETERMINISTIC = True
__C.CUDNN.ENABLE = True
