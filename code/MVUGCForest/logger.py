import logging
import sys
import time
import os

info_dir = "./MVUGCForest_info"
def init_log_dir():
    if not os.path.exists(info_dir):
        os.mkdir(info_dir)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(level = logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s -  %(message)s')
    
    # filename="log/{}.txt".format(time.strftime("%Y%m%d_%H:%M:%S", time.localtime()))
    init_log_dir()
    file_handler = logging.FileHandler(os.path.join(info_dir, "log.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handle=logging.StreamHandler(sys.stderr)
    console_handle.setLevel(logging.INFO)
    console_handle.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handle)

    return logger

def get_opinion_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(level = logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s -  %(message)s')
    
    # filename="log/{}.txt".format(time.strftime("%Y%m%d_%H:%M:%S", time.localtime()))
    init_log_dir()
    file_handler = logging.FileHandler(os.path.join(info_dir,"opinions_log.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

def get_custom_logger(name:str, filename:str, dir:str=None):
    if dir is None:
        dir = info_dir
    mkdir(dir)
    logger = logging.getLogger(name)
    if logger.handlers: # 相同名字的logger只会有一个实例. 如果再构建一个同名的Logger, 会返回之前的Logger实例, 防止重复添加handler
        logger.handlers.pop()

    logger.setLevel(level = logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s -  %(message)s')
    
    file_handler = logging.FileHandler(os.path.join(dir, filename))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

def mkdir(dir:str):
    if not os.path.exists(dir):
        os.mkdir(dir)


# def getmylogger():
#     mylogger = get_custom_logger("mylogger", "logger_script.txt", None)
#     mylogger.info("666")
# if __name__ == "__main__":
#     getmylogger()
#     mylogger = get_custom_logger("mylogger", "logger_script.txt", None)
#     mylogger.info("777")