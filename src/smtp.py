import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from src.logger import Logger


class Smtp:
    """
    Singleton class of the smtp system that will handle the email sending.
    ...

    Attributes
    ----------
    _smtp_instance: Smtp
        Represents the current running instance of Smtp, this will only be created once (by default set to None).
    _print_statements_enabled : bool
        Represents if print statements are going to be enabled (by default set to False).
    """

    _smtp_instance = None
    _print_statements_enabled = False
    _server = None

    @staticmethod
    def instance():
        """
        Obtains instance of Logger.
        """

        return Smtp._smtp_instance

    def __init__(self, _smtp_login, _smtp_password: str) -> None:
        """
        Default constructor.
        """

        if Smtp._smtp_instance is None:
            try:
                # Connect to the SMTP server
                server = smtplib.SMTP('smtp.gmail.com', 587)  # Example: smtp.gmail.com for Gmail
                server.starttls()  # Start TLS encryption
                server.login(_smtp_login, _smtp_password)
                self._server = server

                logging.info(f'::: Smtp successfully authenticated as {_smtp_login}.')
            except Exception as e:
                logging.info(f'::: Smtp authentication failed : {str(e)}')

            Smtp._smtp_instance = self
        else:
            raise Exception("{}: Cannot construct, an instance is already running.".format(__file__))

    def send_email(self, user):
        # Send the email
        msg = MIMEMultipart()
        msg['From'] = self._server.user
        msg['To'] = user["email"]
        msg['Subject'] = "License usage alerting from VoIPtime AMD system!"

        try:
            with open(os.path.join(os.getcwd(), 'email.html'), 'r') as file:
                html_content = file.read()
                html_content = html_content.replace("{email}", user["first_name"] + " " + user["last_name"])
                html_content = html_content.replace("{requests}", str(user["total"] - user["used"]))
            # Attach the message to the MIME
            msg.attach(MIMEText(html_content, 'html'))

            # Send the email
            self._server.sendmail(self._server.user, user["email"], msg.as_string())
            logging.info(f'::: Smtp send warning email to {user["email"]} with request usage.')
        except Exception as e:
            logging.info(f'::: Smtp send email failed : {str(e)} for user {user["email"]}.')
