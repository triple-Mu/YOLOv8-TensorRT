import argparse
from models import EngineBuilder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', help='ONNX file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='TensorRT builder device')
    parser.add_argument(
        '--fp16', action='store_true', help='Build model with fp16 mode')
    args = parser.parse_args()
    return args


def main(args):
    builder = EngineBuilder(args.onnx, args.device)
    builder.build(fp16=args.fp16)


if __name__ == '__main__':
    args = parse_args()
    main(args)
