import numpy as np
import torch
import whisper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE that is used is {DEVICE}")

model = whisper.load_model("large")

print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)

torch.save(
    {
        'model_state_dict': model.state_dict(),
        'dims': model.dims.__dict__,
    },
    '/opt/ml/model/large.pt'
)

print("model saved")

exit(0)