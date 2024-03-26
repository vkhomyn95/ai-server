import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Variables:
    """ This class is responsible for saving and
        loading variables from the system environment.

        Initiated at the entry point
    """

    app_host: str = os.getenv(
        "APP_HOST",
        "127.0.0.1"
    )
    app_port: str = os.getenv(
        "APP_PORT",
        "50053"
    )
    # Directory for storing audio recognition wav files
    audio_dir: str = os.getenv(
        "AUDIO_DIR",
        "/stor/data/audio/"
    )
    # Directory for storing server logs
    logger_dir: str = os.getenv(
        "LOGGER_DIR",
        "/stor/data/logs/server/"
    )
    # Default audio recognition interval
    audio_interval: float = float(os.getenv(
        "DEFAULT_MAX_AUDIO_INTERVAL",
        2.0
    ))
    # Default audio recognition interval
    max_predictions: int = int(os.getenv(
        "DEFAULT_MAX_PREDICTIONS",
        2
    ))
    # Default recognition prediction criteria
    prediction_criteria: str = os.getenv(
        "DEFAULT_PREDICTION_CRITERIA",
        '{"1_interval_1": "True", "1_interval_2": "True", "1_result_3": "True",'
        ' "2_interval_1": "True", "2_interval_2": "False", "2_result_3": "True",'
        ' "3_interval_1": "False", "3_interval_2": "True", "3_result_3": "True",'
        ' "4_interval_1": "False", "4_interval_2": "False", "4_result_3": "False"}'
    )
    # Default audio sample rate
    audio_sample_rate: int = int(os.getenv(
        "DEFAULT_AUDIO_RATE",
        8000
    ))
    # Interval of silence length to be excluded before recognition
    audio_silence_exclude_interval: float = float(os.getenv(
        "DEFAULT_AUDIO_SILENCE_EXCLUDE_INTERVAL",
        0.4
    ))
    audio_silence_coverage_chunks: bytearray = bytearray(
        os.getenv(
            "DEFAULT_AUDIO_SILENCE_COVERAGE_LENGTH",
            int(audio_silence_exclude_interval * audio_sample_rate)
        )
    )

    # Database connection properties
    database_user: str = os.getenv("DATABASE_USER", "root")
    database_password: str = os.getenv("DATABASE_PASSWORD", "root")
    database_host: str = os.getenv("DATABASE_HOST", "127.0.0.1")
    database_port: int = int(os.getenv("DATABASE_PORT", 3306))
    database_name: str = os.getenv("DATABASE_NAME", "amd")

    # Smtp
    smtp_login: str = os.getenv("SMTP_LOGIN", "auth@voiptime.net")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "h9z1keQp9nROb1cdngSYmxUrU")
