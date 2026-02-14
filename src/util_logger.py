
#util_logger.py
import os
import logging
import logging.config
from datetime import date
from datetime import datetime

#from settings import LOGS_PATH, LOG_LEVEL
#from utilities.db_logger_handler import DBLoggingHandler
#"class": "utilities.db_logger_handler.DBLoggingHandler",

LOGS_PATH = "../logs_dir/"
LOG_LEVEL = "DEBUG"#"INFO" #https://docs.python.org/3/library/logging.html#logging.Logger.setLevel
os.makedirs(LOGS_PATH, exist_ok=True)


def setup_logger(module_name=None): #, folder_name=None):
    """
    Desc:
        - 
    """

    dt_time_now = datetime.now()
    hour_now = dt_time_now.strftime("_%m_%d_%Y_%H")  # Date Format == _%m_%d_%Y 
    
    if not os.path.exists(LOGS_PATH):
        os.makedirs(LOGS_PATH)
    # hourly_log_path = os.path.join(LOGS_PATH, f'{hour_now}')
    # if not os.path.exists(hourly_log_path):
    #     os.makedirs(hourly_log_path)
    
    # if folder_name is None:
    #     module_log_file = os.path.join(hourly_log_path, f'{module_name}.log')
    # else:
        # if not os.path.exists(os.path.join(hourly_log_path, module_name)):
        #     os.makedirs(os.path.join(hourly_log_path, module_name))
    if module_name:
        module_log_file_name = "ipwebcam_log_"+hour_now+"00h_" 
        module_log_file = os.path.join(LOGS_PATH, f'{module_log_file_name}.log')

    cnfg_dict = {
        'version': 1,
        "disable_existing_loggers": False,
        "formatters": {
            "ollama_log_format": {
                "format": "%(asctime)s - %(levelname)s -_Name: %(filename)s -_Meth_Name: "
                          "%(funcName)s() -_Line: %(lineno)d -_Log_Message:  %(message)s",
                "datefmt": "_%m_%d_%Y_%H:%M:%S"
            }
        },
        'handlers': {
            'common_handler': {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "level": LOG_LEVEL,
                "formatter": "ollama_log_format",
                "filename": module_log_file,
                "when": "midnight",
                "interval": 1,
                "backupCount": 31,
                "encoding": "utf8"
            }
        
        },
        'loggers': {
            'general': {
                'level': LOG_LEVEL,
                'handlers': ['common_handler']
            }
        }
    }
    if module_name != 'general':
        
        module_handler = {
            'class': 'logging.FileHandler',
            'level': LOG_LEVEL,
            'formatter': "ollama_log_format",
            'filename': module_log_file
        }
        module_logger = {
            'level': LOG_LEVEL,
            'handlers': [f'console_for_{module_name}','common_handler'] #
        }
        if f'console_for_{module_name}' not in cnfg_dict['handlers'].keys():
            cnfg_dict['handlers'][f'console_for_{module_name}'] = module_handler
        if module_name not in cnfg_dict['loggers'].keys():
            cnfg_dict['loggers'][module_name] = module_logger

    logging.config.dictConfig(cnfg_dict)
    return logging.getLogger(module_name)

