from haxllm.model.quantize import QConfig, QuantMethod, QuantSource
from haxllm.model.llama import (
    remap_llama_state_dict,
    remap_qwen_state_dict,
    remap_chatglm_state_dict,
    remap_internlm2_state_dict,
    remap_phi3_state_dict,
    quantize_llama_to_q8,
    convert_llama_q_params,
)
from haxllm.model.gemma import remap_gemma_state_dict
from haxllm.model.utils import get_module

REMAP_FN = {
    "llama": remap_llama_state_dict,
    "qwen": remap_qwen_state_dict,
    "chatglm": remap_chatglm_state_dict,
    "internlm2": remap_internlm2_state_dict,
    "phi3": remap_phi3_state_dict,
    "gemma": remap_gemma_state_dict,
}


def remap_state_dict(*args, **kwargs):
    format = kwargs.pop("format", "llama")
    qconfig: QConfig = kwargs.pop("qconfig", None)
    if format != "llama" and qconfig is not None:
        assert qconfig.source == QuantSource.half and qconfig.method == QuantMethod.rtn_q8_0
    root = REMAP_FN[format](*args, **kwargs)
    if qconfig is not None:
        q_source = qconfig.source
        q_method = qconfig.method
        if q_source == QuantSource.half and q_method == QuantMethod.rtn_q8_0:
            return quantize_llama_to_q8(root, qconfig)
        elif q_source in [QuantSource.autogptq_q4, QuantSource.autoawq_q4, QuantSource.autogptq_q8]:
            return convert_llama_q_params(root, qconfig=qconfig)
        else:
            raise NotImplementedError(f"Quant method {q_method} is not supported for {format} from {q_source}")
    return root
