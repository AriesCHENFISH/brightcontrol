import torch
print("1")
from diffusers import StableDiffusionPipeline
print("1")
pipe = StableDiffusionPipeline.from_pretrained(
    "/data1/cx/sd15",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    use_safetensors=True,
    local_files_only=True,
)
print("1")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
print("1")
print("pipeline load ok")