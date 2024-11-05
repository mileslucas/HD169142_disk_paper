from logging.config import dictConfig

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "console": {
            "format": "[%(asctime)s] %(name)s %(levelname)s %(message)s",
            "datefmt": "H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "console",
        },
    },
    "loggers": {"": {"handlers": ["console"], "level": "INFO", "propagate": True}},
}

dictConfig(LOGGING_CONFIG)
