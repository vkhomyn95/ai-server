import asyncio
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
from src.application import VoicemailRecognitionApplication


class SpeechToTextServicer(stt_pb2_grpc.SpeechToTextServicer):

    def __init__(self, _app: VoicemailRecognitionApplication):
        self.app = _app

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

    async def StreamingRecognize(
            self,
            request_iterator: Iterable[stt_pb2.StreamingRecognizeRequest],
            context
    ):
        """ Main rpc function for converting speech to text using bidirectional stream

            Takes in a stream of stt_pb2 RecognitionAudio messages
            and returns a stream of stt_pb2 StreamingRecognizeResponse messages
        """

        request_id = str(uuid.uuid4())
        logging.info(f'== Request {request_id} received from peer {context.peer()}')

        # extract token from metadata and do validation for current request
        metadata = dict(context.invocation_metadata())

        if metadata is not None:
            token = metadata['authorization']

            # Load authentication principle
            authentication = self.app.auth.is_valid_bearer_token(token, request_id)

            # Validate authentication principle
            if not authentication:
                logging.error(f'== Request {request_id} authorization failed using bearer {token}')
                await context.abort(
                    grpc.StatusCode.PERMISSION_DENIED,
                    "Authorization failed using bearer {}".format(token)
                )

            await asyncio.create_task(self.__split_stream(request_iterator, context, request_id, authentication))

        else:
            logging.error(f'== Request {request_id} authorization failed for None metadata')
            await context.abort(
                grpc.StatusCode.PERMISSION_DENIED, "Authorization failed"
            )

    async def __split_stream(self, request_iterator, context, request_id, authentication):
        """ Place the items from the request_iterator into each
            queue in the list of queues. When using VAD (continuous
            = True), the end-of-speech (EOS) can occur when the
            stream ends or inactivity is detected, whichever occurs
            first.
        """

        await context.send_initial_metadata(
            [
                ("x-request-id", request_id)
            ]
        )

        # Create the directory if it doesn't exist
        os.makedirs(self.app.audio_dir, exist_ok=True)

        # variables to be initialized per request
        config = {}
        predictions = []
        audio_chunks = b''
        start = time.time()
        max_predictions = 2
        num_received_samples = 0
        prediction_criteria = None
        enable_interim_results = True
        sample_rate_hertz = self.app.audio_sample_rate
        desired_num_samples = sample_rate_hertz * self.app.audio_interval
        # end variables to be initialized per request

        async for chunk in request_iterator:
            # extract config from first request iterator
            if not config:
                config = self.app.util.parse_config_from_request(chunk.streaming_config)
                max_predictions = config['interim_results_config']['max_predictions']
                enable_interim_results = config['interim_results_config']['enable_interim_results']
                desired_num_samples = config['sample_rate_hertz'] * config['interim_results_config']['interval']
                # if enabled interim. Then do recognition using criteria condition
                if enable_interim_results:
                    prediction_criteria = self.app.util.parse_config_prediction_criteria(
                        config['interim_results_config']['prediction_criteria'],
                        max_predictions
                    )

            # fill audio variable with bytes content
            predicted = None
            file_name = None
            audio_chunks += chunk.audio_content
            num_received_samples += len(chunk.audio_content)

            if num_received_samples >= desired_num_samples:
                filtered_chunks = self.app.util.swap_zero_bytes(self.app.audio_silence_coverage_chunks, audio_chunks)

                num_filtered_samples = len(filtered_chunks)

                # if audio variable contains desired number of audio chunks
                if num_filtered_samples >= desired_num_samples:
                    file_name = str(uuid.uuid4()) + ".wav"
                    file_path = os.path.join(self.app.audio_dir, file_name)

                    with wave.open(file_path, 'wb') as wf:
                        wf.setnchannels(1)  # Mono audio
                        wf.setsampwidth(2)  # 16-bit audio
                        wf.setframerate(config['sample_rate_hertz'])  # Sample rate
                        wf.writeframes(audio_chunks)

                    predicted = self.app.util.predict_sound_from_bytes(file_path)
                    if not self.app.util.is_ring_condition(predicted):
                        predictions.append(predicted)

                    logging.info(
                        f' == Request {request_id} successfully predicted with {predicted} and stored to {file_path}'
                    )
                    num_received_samples = 0
                    audio_chunks = b''
                # if audio variable contains zero number of audio chunks
                elif num_filtered_samples == 0:
                    logging.info(
                        f' == Request {request_id} unsuccessfully predicted because zero samples received'
                    )
                    num_received_samples = 0
                    audio_chunks = b''
                # if audio variable contains less than required number of audio chunks
                else:
                    logging.info(
                        f' == Request {request_id} unsuccessfully predicted because received {num_filtered_samples}'
                        f' through required {desired_num_samples} audio samples'
                    )
                    audio_chunks = filtered_chunks
                    num_received_samples = num_received_samples - (num_received_samples - num_filtered_samples)

                finish = time.time()

                response = stt_pb2.StreamingRecognizeResponse()
                recognition_result = stt_pb2.SpeechRecognitionResult(
                    start_time=duration_pb2.Duration(
                        seconds=int(start),
                        nanos=int((start - int(start)) * 10**9)
                    ),
                    end_time=duration_pb2.Duration(
                        seconds=int(finish),
                        nanos=int((finish - int(finish)) * 10**9)
                    )
                )
                predictions_count_reached = len(predictions) >= max_predictions

                # If interim result enabled and prediction count was reached than define final result
                if enable_interim_results and predictions_count_reached:
                    transcript, confidence = self.app.util.build_schema_condition(predictions, prediction_criteria)
                    logging.info(
                        f'  == Request {request_id} predicted schema was build as {transcript}.'
                        f' Total recognition time elapsed {finish - start}'
                    )
                # If not interim result enabled than return interim result
                else:
                    transcript, confidence = self.app.util.build_interim_condition(predicted)
                    logging.info(
                        f'  == Request {request_id} predicted as {transcript}.'
                        f' Total recognition time elapsed {finish - start}'
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
                    request_uuid=request_id
                )
                response.results.append(res)

                # Store recognition result
                self.app.db.insert_recognition(
                    predictions_count_reached,
                    request_id,
                    file_name,
                    confidence,
                    transcript,
                    config["extension"],
                    authentication["id"]
                )

                await context.write(response)

                if predictions_count_reached:
                    self.app.db.increment_tariff(
                        authentication["tariff_id"],
                        int(len(predictions) * config['interim_results_config']['interval'])
                    )
                    return
                await asyncio.sleep(1)


async def serve():
    """ Execute the coroutine server and return the result.

        Asynchronous requests are managed
    """

    server = grpc.aio.server(
        futures.ThreadPoolExecutor(
            max_workers=1
        )
    )
    app = VoicemailRecognitionApplication()
    stt_pb2_grpc.add_SpeechToTextServicer_to_server(
        SpeechToTextServicer(
            app
        ),
        server
    )
    server.add_insecure_port("{}:{}".format(app.app_host, app.app_port))

    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
