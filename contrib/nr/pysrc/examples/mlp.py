import psycopg2
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.parse_sql import parse_conditions

# Database connection parameters
db_params = {
    'dbname': 'postgres',
    'user': 'postgres',
    'host': 'localhost',
    'port': '5432'
}


def connect_to_db(db_params):
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    return conn, cursor


def read_data(conn, table: str, where_condition: str, label: str):
    if where_condition != "":
        cond = parse_conditions(where_condition)
        query = f"SELECT * FROM {table} WHERE {cond}"
    else:
        query = f"SELECT * FROM {table}"
    df = pd.read_sql(query, conn)
    features = df.columns.tolist()
    features.remove(label)
    X = df[features].values
    y, _ = pd.factorize(df[label].values)

    num_classes = len(pd.unique(y))  # Ensure this is calculated after factorization

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_test, y_train, y_test, num_classes


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


def train_model(X_train, y_train, num_classes, num_epochs=1000):
    model = MLP(input_size=X_train.shape[1], num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    return model


def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).argmax(dim=1)
        accuracy = accuracy_score(y_test.numpy(), y_pred.numpy())
    return f"Accuracy: {accuracy * 100:.2f}%"


def save_model(model, model_repo: str):
    model_path = model_repo + "/mlp_model.pt"
    torch.jit.save(torch.jit.script(model), model_path)
    with open(model_path, 'rb') as file:
        model_binary = file.read()
    return model_binary, model_path


def create_model_table(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model (
            model_id SERIAL PRIMARY KEY,
            model_name TEXT NOT NULL,
            model_path TEXT,
            model_byte BYTEA
        )
    """)


def insert_model(cursor, model_path, model_binary):
    cursor.execute(
        "INSERT INTO model (model_name, model_path, model_byte) VALUES (%s, %s, %s)",
        ('MLP', model_path, psycopg2.Binary(model_binary))
    )


def close_db(conn, cursor):
    cursor.close()
    conn.close()


def run(table: str, where_condition: str, label: str, model_repo: str) -> dict:
    conn, cursor = connect_to_db(db_params)
    X_train, X_test, y_train, y_test, num_classes = read_data(conn, table, where_condition, label)
    model = train_model(X_train, y_train, num_classes)
    res = evaluate_model(model, X_test, y_test)
    model_binary, model_path = save_model(model, model_repo)
    create_model_table(cursor)
    insert_model(cursor, model_path, model_binary)
    conn.commit()
    close_db(conn, cursor)
    return res
