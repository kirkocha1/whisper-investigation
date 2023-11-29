import numpy as np
import torch
import whisper
import sagemaker


def save_model():

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
        '/home/kochango/workplace/WhisperService/local_test/test_dir/model/whisper.pt'
    )
    print("model saved")

def load_model():
    sess = sagemaker.session.Session()
    bucket = sess.default_bucket()
    prefix = f'whisper-deploy/'
    s3_uri = f's3://{bucket}/{prefix}'
    model_uri = sess.upload_data('test_dir/model/model.tar.gz', bucket = bucket, key_prefix=f"{prefix}model")
    print(f"{s3_uri}model/model.tar.gz")

save_model()
load_model()
exit(0)