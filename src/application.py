from dataclasses import dataclass
import os

from src.auth import VoicemailRecognitionAuthenticator
from src.database import VoicemailRecognitionDatabase
from src.logger import Logger
from src.audioutils import VoicemailRecognitionAudioUtil


@dataclass(frozen=True)
class VoicemailRecognitionApplication:
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
        "/stor/data/logs/"
    )
    # Default audio recognition interval
    audio_interval: float = float(os.getenv(
        "DEFAULT_AUDIO_INTERVAL",
        2.0
    ))
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
    database_name: str = os.getenv("DATABASE_NAME", "speecher")

    # Class initialization
    logger = Logger(logger_dir)
    db = VoicemailRecognitionDatabase(
        database_user,
        database_password,
        database_host,
        database_port,
        database_name
    )
    auth = VoicemailRecognitionAuthenticator(db.instance())
    util = VoicemailRecognitionAudioUtil()

    # db.insert_user(
    #     "Volodymyr",
    #     "Khomyn",
    #     "vkhomyn@viptime.net",
    #     "0984154969",
    #     "voiptime.net",
    #     True,
    #     1000,
    #     True,
    #     1000
    # )
