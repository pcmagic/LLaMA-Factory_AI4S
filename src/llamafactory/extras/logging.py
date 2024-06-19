import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from .constants import RUNNING_LOG


class LoggerHandler(logging.Handler):
    r"""
    Logger handler used in Web UI.
    """

    def __init__(self, output_dir: str) -> None:
        super().__init__()
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
        )
        self.setLevel(logging.INFO)
        self.setFormatter(formatter)

        os.makedirs(output_dir, exist_ok=True)
        self.running_log = os.path.join(output_dir, RUNNING_LOG)
        if os.path.exists(self.running_log):
            os.remove(self.running_log)

        self.thread_pool = ThreadPoolExecutor(max_workers=1)

    def _write_log(self, log_entry: str) -> None:
        with open(self.running_log, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n\n")

    def emit(self, record) -> None:
        if record.name == "httpx":
            return

        log_entry = self.format(record)
        self.thread_pool.submit(self._write_log, log_entry)

    def close(self) -> None:
        self.thread_pool.shutdown(wait=True)
        return super().close()


def get_logger(name: str) -> logging.Logger:
    r"""
    Gets a standard logger with a stream hander to stdout.
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
    )

    # 创建一个handler，用于输出到控制台
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)  # set each log_handler separately.

    # 创建一个handler，用于写入日志文件
    import os
    logger_path = os.environ.get('OUTPUT_DIR', None)
    # logger_path = os.path.join(os.getcwd(), 'llamafactory_log') if logger_path is None \
    #     else os.path.join(logger_path, 'llamafactory_log')
    # if not os.path.exists(logger_path):
    #     os.makedirs(logger_path)
    logger_path = os.getcwd() if logger_path is None else logger_path
    logger_name = os.path.join(logger_path,
                               "llamafactory.log" if not name.endswith('.log') else name)
    file_handler = logging.FileHandler(logger_name)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # set each log_handler separately.

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.info("saving log info to file %s" % logger_name)

    return logger


def reset_logging() -> None:
    r"""
    Removes basic config of root logger. (unused in script)
    """
    root = logging.getLogger()
    list(map(root.removeHandler, root.handlers))
    list(map(root.removeFilter, root.filters))
