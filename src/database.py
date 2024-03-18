import datetime
import logging
import re
import sys

import mariadb
from mariadb import OperationalError, ProgrammingError


class Database:
    """
        Singleton class of the logger system that will handle the logging system.
    ...

    Attributes
    ----------
    _database_instance: Database
        Represents the current running instance of VoicemailRecognitionDatabase,
        this will only be created once (by default set to None).

    """

    _database_instance = None

    @staticmethod
    def instance():
        """
            Obtains instance of VoicemailRecognitionDatabase.
        """

        return Database._database_instance

    def __init__(self, _user: str, _password: str, _host: str, _port: int, _database: str) -> None:
        """
            Default constructor.
        """

        if Database._database_instance is None:
            try:
                conn = mariadb.connect(
                    user=_user,
                    password=_password,
                    host=_host,
                    port=_port,
                    database=_database,
                    autocommit=True,
                    reconnect=True
                )
                Database._database_instance = self
            except mariadb.Error as e:
                logging.error(f'  >> Error connecting to MariaDB Platform: {e}')
                sys.exit(1)
            conn.auto_reconnect = True
            self.conn = conn
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
        try:
            current_date = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')
            self.cur.execute(
                f'INSERT into recognition '
                f'(created_date, final, request_uuid, audio_uuid, confidence, prediction, extension, user_id) '
                f'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (current_date, final, request_uuid, audio_uuid, confidence * 100, prediction, extension, user_id,)
            )
        except mariadb.InterfaceError as e:
            logging.error(f'  >> Error connecting to MariaDB Platform: {e}')
            self.conn.reconnect()

    def load_user_by_api_key(self, api_key: str):
        try:
            self.cur.execute(
                f'SELECT * from user u '
                f'left join tariff t on t.id=u.tariff_id '
                f'left join recognition_configuration c on c.id=u.recognition_id '
                f'where u.api_key=?',
                (api_key,)
            )
            return self.cur.fetchone()
        except mariadb.InterfaceError as e:
            logging.error(f'  >> Error connecting to MariaDB Platform: {e}')
            self.conn.reconnect()
            return self.load_user_by_api_key(api_key)

    def increment_tariff(self, tariff_id):
        try:
            self.cur.execute(
                f'UPDATE tariff SET used = used + 1 where id = ?',
                (tariff_id,)
            )
        except mariadb.InterfaceError as e:
            logging.error(f'  >> Error connecting to MariaDB Platform: {e}')
            self.conn.reconnect()
            return self.increment_tariff(tariff_id)