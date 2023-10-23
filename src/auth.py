import logging

import jwt

from src.database import VoicemailRecognitionDatabase


class VoicemailRecognitionAuthenticator:
    """ This class is responsible for inspecting metadata
        received in jwt token.

        Initiated at the entry point
    """

    def __init__(self, db: VoicemailRecognitionDatabase):
        self.db = db

    def is_valid_bearer_token(self, bearer_token, request_id):
        """
            Validate token received in metadata
        """

        bearer = bearer_token.removeprefix("Bearer ")

        header_data = jwt.get_unverified_header(bearer)
        body_data = jwt.decode(bearer, options={"verify_signature": False})

        if header_data:
            authentication = self.db.load_user_by_api_key(header_data["kid"])
            if authentication is not None:
                # Check request in tariff
                if bool(authentication["request"]):
                    if authentication["request_size"] > authentication["request_limit"]:
                        logging.error(f'== Request {request_id} authorization failed from {body_data["aud"]}'
                                      f' because request limit reached for user id={authentication["id"]}')
                        return None
                # Check audio in tariff
                if bool(authentication["audio"]):
                    if authentication["audio_size"] > authentication["audio_limit"]:
                        logging.error(f'== Request {request_id} authorization failed from {body_data["aud"]}'
                                      f' because audio limit reached for user id={authentication["id"]}')
                        return None
                return authentication

        return None