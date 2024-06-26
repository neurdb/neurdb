# main script for PoC experiment, inference running in python environment
import time
import torch
from torch import nn
import argparse

from prepare_data import connect_to_db


BATCH_SIZE = 10
MODEL_PATH = 'mlp_model.pt'


class MLP(nn.Module):
    def __init__(self, input_size=4, hidden_size=10, num_classes=3):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = torch.relu(self.hidden(x))
        x = torch.softmax(self.output(x), dim=1)
        return x


def load_model():
    model = torch.jit.load(MODEL_PATH)
    model.eval()
    return model


def query_data():
    conn, cursor = connect_to_db()
    cursor.execute('SELECT * FROM iris')
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def inference(rows: list, batch_size: int = 10):
    # group rows into batches
    batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
    model = load_model()
    for batch in batches:
        data = torch.tensor([row[:-1] for row in batch], dtype=torch.float32)
        predictions = model(data)


class Logger:
    def __init__(self):
        self.start_time = time.monotonic()
        self.end_time = 0
        self.duration = 0
        self.records = []

    def start(self, msg):
        record = self.LogRecord(msg)
        record.start()
        self.records.append(record)

    def end(self):
        record = self.records[-1]
        record.end()

    def print_records(self):
        for record in self.records:
            print(record)

    def end_all(self):
        self.end_time = time.monotonic()
        self.duration = (self.end_time - self.start_time) * 1000
        print(f'Total duration: {self.duration} ms')
        self.print_records()

    class LogRecord:
        def __init__(self, msg):
            self.msg = msg
            self.start_time = 0
            self.end_time = 0
            self.duration = 0  # in milliseconds

        def start(self):
            self.start_time = time.monotonic()

        def end(self):
            self.end_time = time.monotonic()
            self.duration = (self.end_time - self.start_time) * 1000

        def __str__(self):
            return f'{self.msg}: {self.duration} ms'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference on the iris dataset')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for inference')
    parser.add_argument('--model_path', type=str, default='mlp_model.pt', help='Path to the model')
    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    MODEL_PATH = args.model_path

    logger = Logger()
    logger.start('query data from database')
    rows = query_data()
    logger.end()
    logger.start('inference')
    inference(rows, BATCH_SIZE)
    logger.end()
    logger.end_all()
