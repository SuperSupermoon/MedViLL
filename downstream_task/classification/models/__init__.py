from models.model import MultimodalBertClf


MODELS = {
    "model": MultimodalBertClf,
}

def get_model(args):
    return MODELS['model'](args)