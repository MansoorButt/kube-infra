import socket
import pickle
import threading
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import struct
import os
import logging
import sys
from datetime import datetime

# Configure logging
log_dir = '/app/logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'client_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Get server details from environment variables
SERVER_HOST = os.getenv('SERVER_HOST', '0.0.0.0')
SERVER_PORT = int(os.getenv('SERVER_PORT', '58000'))

null_model = None
trained_model = None
null_model_received = False

logger.info(f"Attempting to connect to server at {SERVER_HOST}:{SERVER_PORT}")
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('fl-server-service', 58000))
logger.info("Successfully connected to server")

def train_model(model):
    logger.info("Starting model training")
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    logger.info(f"Model training complete. Test accuracy: {score:.4f}")
    return model

def send_trained_model():
    global null_model_received
    
    while not null_model_received:
        threading.Event().wait(1)
    
    try:
        logger.info("Starting model training process")
        trained_model = train_model(null_model)
        serialized = pickle.dumps(trained_model)
        
        # Send size
        size = len(serialized)
        logger.info(f"Sending trained model (size: {size} bytes)")
        client.sendall(struct.pack('!I', size))
        
        # Send model
        client.sendall(serialized)
        
        # Wait for acknowledgment
        response = client.recv(1024).decode('utf-8').strip()
        if response == "MODEL_RECEIVED":
            logger.info("Server acknowledged model reception")
        else:
            logger.warning(f"Unexpected server response: {response}")
        
    except Exception as e:
        logger.error(f"Error in sending trained model: {e}")

def receive_null_model():
    global null_model, null_model_received
    
    try:
        while True:
            header = b''
            while b'\n' not in header:
                chunk = client.recv(1)
                if not chunk:
                    return
                header += chunk
            
            header = header.decode('utf-8').strip()
            
            if header.startswith("MODEL:"):
                size = int(header.split(":")[1])
                logger.info(f"Receiving null model of size {size} bytes")
                data = b''
                
                while len(data) < size:
                    chunk = client.recv(min(size - len(data), 4096))
                    if not chunk:
                        break
                    data += chunk
                
                null_model = pickle.loads(data)
                null_model_received = True
                logger.info("Successfully received null model")
                break
                
            elif header.startswith("MESSAGE:"):
                message = header.split(":", 1)[1]
                logger.info(f"Server message: {message}")
                
    except Exception as e:
        logger.error(f"Error receiving data: {e}")

logger.info("Starting client threads")
receive_thread = threading.Thread(target=receive_null_model)
receive_thread.start()

send_thread = threading.Thread(target=send_trained_model)
send_thread.start()

receive_thread.join()
send_thread.join()
logger.info("Client shutting down")