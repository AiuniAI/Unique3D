import onnxruntime
import torch

providers = [
    ('TensorrtExecutionProvider', {
        'device_id': 0,
        'trt_max_workspace_size': 8 * 1024 * 1024 * 1024,
        'trt_fp16_enable': True,
        'trt_engine_cache_enable': True,
    }),
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',
        'gpu_mem_limit': 8 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'HEURISTIC',
    })
]

def load_onnx(file_path: str):
    assert file_path.endswith(".onnx")
    sess_opt = onnxruntime.SessionOptions()
    ort_session = onnxruntime.InferenceSession(file_path, sess_opt=sess_opt, providers=providers)
    return ort_session


def load_onnx_caller(file_path: str, single_output=False):
    ort_session = load_onnx(file_path)
    def caller(*args):
        torch_input = isinstance(args[0], torch.Tensor)
        if torch_input:
            torch_input_dtype = args[0].dtype
            torch_input_device = args[0].device
            # check all are torch.Tensor and have same dtype and device
            assert all([isinstance(arg, torch.Tensor) for arg in args]), "All inputs should be torch.Tensor, if first input is torch.Tensor"
            assert all([arg.dtype == torch_input_dtype for arg in args]), "All inputs should have same dtype, if first input is torch.Tensor"
            assert all([arg.device == torch_input_device for arg in args]), "All inputs should have same device, if first input is torch.Tensor"
            args = [arg.cpu().float().numpy() for arg in args]
        
        ort_inputs = {ort_session.get_inputs()[idx].name: args[idx] for idx in range(len(args))}
        ort_outs = ort_session.run(None, ort_inputs)
        
        if torch_input:
            ort_outs = [torch.tensor(ort_out, dtype=torch_input_dtype, device=torch_input_device) for ort_out in ort_outs]
        
        if single_output:
            return ort_outs[0]
        return ort_outs
    return caller
