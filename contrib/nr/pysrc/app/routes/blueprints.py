from flask import Blueprint

train_bp = Blueprint('train', __name__)
inference_bp = Blueprint('inference', __name__)
finetune_bp = Blueprint('finetune', __name__)
context_bp = Blueprint('context', __name__)
