import logging
import os
import time
import uuid
import wave
from concurrent import futures
from typing import Iterable

import grpc
from google.protobuf import duration_pb2

import stt_pb2
import stt_pb2_grpc
from src.audioutils import VoicemailRecognitionAudioUtil
from src.auth import VoicemailRecognitionAuthenticator
from src.database import Database
from src.logger import Logger
from src.variables import Variables


class SpeechToTextServicer(stt_pb2_grpc.SpeechToTextServicer):

    def __init__(
            self,
            _variables: Variables,
            _auth: VoicemailRecognitionAuthenticator,
            _util: VoicemailRecognitionAudioUtil,
            _db: Database
    ):
        self.variables = _variables
        self.auth = _auth
        self.util = _util
        self.db = _db

    def Recognize(
            self,
            request,
            context
    ):
        """ Main rpc function for converting speech to text using unary stream

            Takes in a file of stt_pb2 RecognitionAudio message
            and returns stt_pb2 StreamingRecognizeResponse
        """

        return super().Recognize(request, context)

    def StreamingRecognize(
            self,
            request_iterator: Iterable[stt_pb2.StreamingRecognizeRequest],
            context
    ):
        """ Main rpc function for converting speech to text using bidirectional stream

            Takes in a stream of stt_pb2 RecognitionAudio messages
            and returns a stream of stt_pb2 StreamingRecognizeResponse messages
        """

        # retrieve variables to be initialized per request
        configurations = next(request_iterator).streaming_config

        logging.info(f'== Request {configurations.config.request_uuid} received from peer {context.peer()}')

        # extract token from metadata and do validation for current request
        metadata = dict(context.invocation_metadata())

        if metadata is not None:
            token = metadata['authorization']

            # Load authentication principle
            authentication = self.auth.is_valid_bearer_token(token, configurations.config.request_uuid)

            # Validate authentication principle
            if not authentication:
                logging.error(f'== Request {configurations.config.request_uuid} authorization failed using bearer {token}')
                context.abort(
                    grpc.StatusCode.PERMISSION_DENIED,
                    "Authorization failed using bearer {}".format(token)
                )

            config = self.util.parse_request_config(configurations, authentication)
            # AI AMD stream recognition
            return self.__split_stream_amd(request_iterator, context, authentication, config)

        else:
            logging.error(f'== Request {configurations.config.request_uuid} authorization failed for None metadata')
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED, "Authorization failed"
            )

    def __split_stream_amd(self, request_iterator, context, user, config):
        """ Place the items from the request_iterator into each
            queue in the list of queues. When using VAD (continuous
            = True), the end-of-speech (EOS) can occur when the
            stream ends or inactivity is detected, whichever occurs
            first.
        """

        context.send_initial_metadata(
            [
                ("x-request-id", config["request_id"])
            ]
        )

        predictions = []
        file_chunks = b''
        audio_chunks = b''
        start = time.time()
        num_received_samples = 0
        predictions_count_reached = False

        for chunk in request_iterator:
            predicted = None
            file_name = None
            file_chunks += chunk.audio_content
            audio_chunks += chunk.audio_content
            num_received_samples += len(chunk.audio_content)

            if num_received_samples >= config['desired_num_samples'] and not predictions_count_reached:
                filtered_chunks = self.util.swap_zero_bytes(self.variables.audio_silence_coverage_chunks, audio_chunks)

                num_filtered_samples = len(filtered_chunks)

                # if audio variable contains desired number of audio chunks
                if num_filtered_samples >= config['desired_num_samples']:
                    file_name = str(uuid.uuid4()) + ".wav"
                    file_path = os.path.join(self.util.get_save_directory(), file_name)

                    with wave.open(file_path, 'wb') as wf:
                        wf.setnchannels(1)  # Mono audio
                        wf.setsampwidth(2)  # 16-bit audio
                        wf.setframerate(config['sample_rate_hertz'])  # Sample rate
                        wf.writeframes(audio_chunks)

                    predicted = self.util.predict_sound_from_bytes(file_path)
                    if not self.util.is_ring_condition(predicted):
                        predictions.append(predicted)

                    logging.info(
                        f' == Request {config["request_id"]} successfully predicted with {predicted} and stored to {file_path}'
                    )
                    num_received_samples = 0
                    audio_chunks = b''
                # if audio variable contains zero number of audio chunks
                elif num_filtered_samples == 0:
                    logging.info(
                        f' == Request {config["request_id"]} unsuccessfully predicted because zero samples received'
                    )
                    num_received_samples = 0
                    audio_chunks = b''
                # if audio variable contains less than required number of audio chunks
                else:
                    logging.info(
                        f' == Request {config["request_id"]} unsuccessfully predicted because received {num_filtered_samples}'
                        f' through required {config["desired_num_samples"]} audio samples'
                    )
                    audio_chunks = filtered_chunks
                    num_received_samples = num_received_samples - (num_received_samples - num_filtered_samples)

                finish = time.time()

                response = stt_pb2.StreamingRecognizeResponse()
                recognition_result = stt_pb2.SpeechRecognitionResult(
                    start_time=duration_pb2.Duration(
                        seconds=int(start),
                        nanos=int((start - int(start)) * 10 ** 9)
                    ),
                    end_time=duration_pb2.Duration(
                        seconds=int(finish),
                        nanos=int((finish - int(finish)) * 10 ** 9)
                    )
                )

                transcript, confidence = self.util.build_interim_condition(predicted)

                logging.info(
                    f'  == Request {config["request_id"]} predicted as {transcript}.'
                    f' Total recognition time elapsed {finish - start}'
                )

                # Store recognition result
                self.db.insert_recognition(
                    False,
                    config["request_id"],
                    file_name,
                    confidence,
                    transcript,
                    config["extension"],
                    user["id"],
                    config["company_id"],
                    config["campaign_id"],
                    config["application_id"]
                )

                predictions_count_reached = len(predictions) >= config['max_predictions']

                # If interim result enabled and prediction count was reached than define final result
                if predictions_count_reached:
                    transcript, confidence = self.util.build_schema_condition(
                        predictions, config['prediction_criteria']
                    )

                    logging.info(
                        f'  == Request {config["request_id"]} predicted schema was build as {transcript}.'
                        f' Total recognition time elapsed {finish - start}'
                    )

                    # Store recognition result
                    self.db.insert_recognition(
                        True,
                        config["request_id"],
                        file_name,
                        confidence,
                        transcript,
                        config["extension"],
                        user["id"],
                        config["company_id"],
                        config["campaign_id"],
                        config["application_id"]
                    )

                # Build GRPC session response event
                recognition_result.alternatives.append(
                    stt_pb2.SpeechRecognitionAlternative(
                        transcript=transcript,
                        confidence=confidence
                    )
                )
                res = stt_pb2.StreamingRecognitionResult(
                    recognition_result=recognition_result,
                    is_final=predictions_count_reached,
                    request_uuid=config["request_id"]
                )
                response.results.append(res)

                yield response
            elif predictions_count_reached:
                file_name = config["request_id"] + ".wav"
                file_path = os.path.join(self.util.get_save_directory(), file_name)

                with wave.open(file_path, 'wb') as wf:
                    wf.setnchannels(1)  # Mono audio
                    wf.setsampwidth(2)  # 16-bit audio
                    wf.setframerate(config['sample_rate_hertz'])  # Sample rate
                    wf.writeframes(file_chunks)

                self.db.increment_tariff(user["tariff_id"])
                # left = user["total"] - 1
                # if left == round(user["total"] * 0.1) or left == round(user["total"] * 0.05):
                #     self.smtp.send_email(user)
                yield

        if not predictions_count_reached:
            logging.info(f'  == Request {config["request_id"]} predicted schema was build as not_predicted.')
            # Store recognition result
            self.db.insert_recognition(
                True,
                config["request_id"],
                None,
                1,
                "not_predicted",
                config["extension"],
                user["id"],
                config["company_id"],
                config["campaign_id"],
                config["application_id"]
            )


def serve():
    """ Execute the coroutine server and return the result.

        Asynchronous requests are managed
    """

    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=1
        )
    )

    variables = Variables()
    # Create the directory if it doesn't exist
    os.makedirs(variables.audio_dir, exist_ok=True)
    os.makedirs(variables.logger_dir, exist_ok=True)

    # Class initialization
    logger = Logger(variables.logger_dir)
    # smtp = Smtp(variables.smtp_login, variables.smtp_password)

    db = Database(
        variables.database_user,
        variables.database_password,
        variables.database_host,
        variables.database_port,
        variables.database_name
    )
    auth = VoicemailRecognitionAuthenticator(db.instance())
    util = VoicemailRecognitionAudioUtil(variables)

    stt_pb2_grpc.add_SpeechToTextServicer_to_server(
        SpeechToTextServicer(
            variables, auth, util, db, smtp
        ),
        server
    )
    server.add_insecure_port("{}:{}".format(variables.app_host, variables.app_port))

    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
