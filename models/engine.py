from pathlib import Path
from typing import Optional, Union, List
from collections import namedtuple

try:
    import tensorrt as trt
except Exception:
    trt = None
import warnings
import torch

warnings.filterwarnings(action='ignore', category=DeprecationWarning)


class EngineBuilder:

    def __init__(self, checkpoint: Union[str, Path], device: Optional[Union[str, int, torch.device]] = None) -> None:
        checkpoint = Path(checkpoint) if isinstance(checkpoint, str) else checkpoint
        assert checkpoint.exists() and checkpoint.suffix == '.onnx'
        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device(f'cuda:{device}')

        self.checkpoint = checkpoint
        self.device = device

    def __build_engine(self, fp16: bool = True, with_profiling: bool = True) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace='')
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = torch.cuda.get_device_properties(self.device).total_memory
        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(self.checkpoint)):
            raise RuntimeError(f'failed to load ONNX file: {str(self.checkpoint)}')
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]

        for inp in inputs:
            logger.log(trt.Logger.WARNING, f'input "{inp.name}" with shape{inp.shape} {inp.dtype}')
        for out in outputs:
            logger.log(trt.Logger.WARNING, f'output "{out.name}" with shape{out.shape} {out.dtype}')
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        self.weight = self.checkpoint.with_suffix('.engine')

        if with_profiling:
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        with builder.build_engine(network, config) as engine:
            self.weight.write_bytes(engine.serialize())
        logger.log(trt.Logger.WARNING, f'Build tensorrt engine finish.\nSave in {str(self.weight.absolute())}')

    def build(self, fp16: bool = True, with_profiling=True):
        self.__build_engine(fp16, with_profiling)


class TRTModule(torch.nn.Module):
    dtypeMapping = {trt.bool: torch.bool,
                    trt.int8: torch.int8,
                    trt.int32: torch.int32,
                    trt.float16: torch.float16,
                    trt.float32: torch.float32}

    def __init__(self, weight: Union[str, Path], device: Optional[torch.device]):
        super(TRTModule, self).__init__()
        self.weight = Path(weight) if isinstance(weight, str) else weight
        self.device = device if device is not None else torch.device('cuda:0')
        self.stream = torch.cuda.Stream(device=device)
        self.__init_engine()
        self.__init_bindings()

    def __init_engine(self):
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace='')
        with trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(self.weight.read_bytes())

        context = model.create_execution_context()

        names = [model.get_binding_name(i) for i in range(model.num_bindings)]
        self.num_bindings = model.num_bindings
        self.bindings: List[int] = [0] * self.num_bindings
        num_inputs, num_outputs = 0, 0

        for i in range(model.num_bindings):
            if model.binding_is_input(i):
                num_inputs += 1
            else:
                num_outputs += 1

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model = model
        self.context = context
        self.input_names = names[:num_inputs]
        self.output_names = names[num_inputs:]

    def __init_bindings(self):
        dynamic = False
        Tensor = namedtuple('Tensor', ('name', 'dtype', 'shape'))
        inp_info = []
        out_info = []
        for i, name in enumerate(self.input_names):
            assert self.model.get_binding_name(i) == name
            dtype = self.dtypeMapping[self.model.get_binding_dtype(i)]
            shape = tuple(self.model.get_binding_shape(i))
            if -1 in shape: dynamic = True
            inp_info.append(Tensor(name, dtype, shape))
        for i, name in enumerate(self.output_names):
            i += self.num_inputs
            assert self.model.get_binding_name(i) == name
            dtype = self.dtypeMapping[self.model.get_binding_dtype(i)]
            shape = tuple(self.model.get_binding_shape(i))
            out_info.append(Tensor(name, dtype, shape))

        if not dynamic:
            self.output_tensor = [torch.empty(info.shape, dtype=info.dtype, device=self.device) for info in out_info]
        self.is_dynamic = dynamic

    def forward(self, *inputs):

        assert len(inputs) == self.num_inputs
        contiguous_inputs: List[torch.Tensor] = [i.contiguous() for i in inputs]

        for i in range(self.num_inputs):
            self.bindings[i] = contiguous_inputs[i].data_ptr()
            if self.is_dynamic:
                self.context.set_binding_shape(i, tuple(contiguous_inputs[i].shape))

        outputs: List[torch.Tensor] = []

        for i in range(self.num_outputs):
            j = i + self.num_inputs
            if self.is_dynamic:
                shape = tuple(self.context.get_binding_shape(j))
                output = torch.empty(size=shape, dtype=self.out_info[i].dtype, device=self.device)
            else:
                output = self.output_tensor[i]
            self.bindings[j] = output.data_ptr()
            outputs.append(output)

        self.context.execute_async_v2(self.bindings, self.stream.cuda_stream)
        self.stream.synchronize()

        return tuple(outputs) if len(outputs) > 1 else outputs[0]
