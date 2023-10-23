import datetime
import logging
import re
import secrets
import sys

import mariadb
from mariadb import OperationalError, ProgrammingError


class VoicemailRecognitionDatabase:
    """
        Singleton class of the logger system that will handle the logging system.
    ...

    Attributes
    ----------
    _database_instance: VoicemailRecognitionDatabase
        Represents the current running instance of VoicemailRecognitionDatabase,
        this will only be created once (by default set to None).

    """

    _database_instance = None

    @staticmethod
    def instance():
        """
            Obtains instance of VoicemailRecognitionDatabase.
        """

        return VoicemailRecognitionDatabase._database_instance

    def __init__(self, _user: str, _password: str, _host: str, _port: int, _database: str) -> None:
        """
            Default constructor.
        """

        if VoicemailRecognitionDatabase._database_instance is None:
            try:
                conn = mariadb.connect(
                    user=_user,
                    password=_password,
                    host=_host,
                    port=_port,
                    database=_database,
                    autocommit=True
                )
                VoicemailRecognitionDatabase._database_instance = self
            except mariadb.Error as e:
                logging.error(f'  >> Error connecting to MariaDB Platform: {e}')
                sys.exit(1)

            self.cur = conn.cursor(dictionary=True)

            # Run migrations
            self.exec_sql_file("db.sql")

            # self.cur = conn.cursor()
            # self.curd = conn.cursor(dictionary=True)

        else:
            raise Exception("{}: Cannot construct, an instance is already running.".format(__file__))

    def exec_sql_file(self, sql_file):
        logging.info(f'  >> Executing SQL script file: {sql_file}')
        statement = ""

        for line in open(sql_file):
            if re.match(r'--', line):  # ignore sql comment lines
                continue
            if not re.search(r';$', line):  # keep appending lines that don't end in ';'
                statement = statement + line
            else:  # when you get a line ending in ';' then exec statement and reset for next statement
                statement = statement + line
                logging.debug(f'  >> Executing SQL statement:\n {statement}')
                try:
                    self.cur.execute(statement)
                except (OperationalError, ProgrammingError) as e:
                    logging.warning(f'  >> MySQLError during execute statement: {e.args}')
                statement = ""

    def insert_recognition(
            self,
            final: bool,
            request_uuid: str,
            audio_uuid: str,
            confidence: float,
            prediction: str,
            extension: str,
            user_id: int
    ):
        current_date = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')
        self.cur.execute(
            f'INSERT into recognition '
            f'(created_date, final, request_uuid, audio_uuid, confidence, prediction, extension, user_id) '
            f'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (current_date, final, request_uuid, audio_uuid, confidence * 100, prediction, extension, user_id,)
        )

    def insert_user(
            self,
            first_name: str,
            last_name: str,
            email: str,
            phone: str,
            audience: str,
            request: bool,
            request_limit: int,
            audio: bool,
            audio_limit: int
    ):
        if not email:
            logging.error(f'  >> MySQLError during execute statement: email can not be null')
        elif not phone:
            logging.error(f'  >> MySQLError during execute statement: phone can not be null')

        current_date = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')

        self.cur.execute(
            f'INSERT into tariff '
            f'(created_date, updated_date, request, request_limit, audio, audio_limit) '
            f'VALUES (?, ?, ?, ?, ?, ?)',
            (current_date, current_date, request, request_limit, audio, audio_limit,)
        )

        tariff_id = self.cur.lastrowid
        api_key = secrets.token_urlsafe(32)
        secret_key = secrets.token_urlsafe(32)

        self.cur.execute(
            f'INSERT into user '
            f'(created_date,updated_date,first_name,last_name,email,phone,api_key,secret_key,audience,tariff_id) '
            f'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (current_date, current_date, first_name, last_name, email, phone, api_key, secret_key, audience, tariff_id,)
        )
        return self.cur.lastrowid

    def load_user_by_id(self, user_id: int):
        self.cur.execute(
            f'SELECT * from user u left join tariff t on t.id=u.tariff_id where u.id=?',
            (user_id,)
        )
        return self.cur.fetchone()

    def load_user_by_api_key(self, api_key: str):
        self.cur.execute(
            f'SELECT * from user u left join tariff t on t.id=u.tariff_id where u.api_key=?',
            (api_key,)
        )
        return self.cur.fetchone()

    def increment_tariff(self, tariff_id, audio_size):
        self.cur.execute(
            f'UPDATE tariff SET request_size = request_size + 1, audio_size = audio_size + ?  where id = ?',
            (audio_size,  tariff_id,)
        )
