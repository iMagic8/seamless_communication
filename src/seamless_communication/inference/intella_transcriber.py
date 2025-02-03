import torch
import logging
from seamless_communication.inference import Transcriber

# === Option 1: Set up logging filter ===
class ReplaceModelNameFilter(logging.Filter):
    def filter(self, record):
        if isinstance(record.msg, str):
            record.msg = record.msg.replace("seamlessM4T_v2_large", "MyCustomModelName")
        return True

# Attach filter to the library's logger (adjust the logger name if needed)
logger = logging.getLogger("seamless_communication")
logger.addFilter(ReplaceModelNameFilter())
logging.getLogger().addFilter(ReplaceModelNameFilter())

# === Option 2: Alternatively, override print (comment out if not needed) ===
# import builtins
# original_print = print
# def custom_print(*args, **kwargs):
#     new_args = [
#         arg.replace("seamlessM4T_v2_large", "MyCustomModelName") if isinstance(arg, str) else arg 
#         for arg in args
#     ]
#     original_print(*new_args, **kwargs)
# builtins.print = custom_print

# Check for CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32

AUDIO_SAMPLE_RATE = 16000
target_language_code = "arb"

# Initialize Transcriber (the output here should now display "MyCustomModelName" instead of the original model name)
transcriber = Transcriber(
    "seamlessM4T_v2_large",  # the library will still load the correct model, but the output text will be replaced
    device=device,
    dtype=dtype,
)
