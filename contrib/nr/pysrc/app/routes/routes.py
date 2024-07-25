from flask import Blueprint

train_bp = Blueprint('train', __name__)
inference_bp = Blueprint('inference', __name__)
finetune_bp = Blueprint('finetune', __name__)

