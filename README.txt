=================================================
Action to generate stt_pb2.py and stt_pb2_grpc.py
=================================================

pip3 install googleapis-common-protos

python3 -m grpc_tools.protoc -I protos --python_out=. --grpc_python_out=. protos/stt.proto