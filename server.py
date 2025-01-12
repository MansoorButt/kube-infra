import socket
import pickle
import threading
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import struct
import sys
import time
import logging
import os
from datetime import datetime

# Configure logging
log_dir = '/app/logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'server_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Global variables
model_null = RandomForestClassifier()
data = load_iris()
X, y = data.data, data.target

# Global flags
is_training_complete = False
broadcast_to_all_clients = False
null_model_sent = False
should_exit = False

host = '0.0.0.0'
port = 58000

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((host, port))
server.listen(5)

clients = {}
models = []
received_models = set()

def close_all_connections():
    global should_exit
    logger.info("Initiating shutdown sequence")
    for client_socket in clients.keys():
        try:
            client_socket.close()
            logger.info("Closed client connection")
        except Exception as e:
            logger.error(f"Error closing client connection: {e}")
    
    try:
        server.close()
        logger.info("Closed server socket")
    except Exception as e:
        logger.error(f"Error closing server socket: {e}")
    
    logger.info("All connections closed")
    should_exit = True

def broadcast_message(message):
    logger.info(f"Broadcasting message to all clients: {message}")
    for client_socket in clients.keys():
        try:
            header = f"MESSAGE:{message}\n".encode('utf-8')
            client_socket.sendall(header)
        except Exception as e:
            logger.error(f"Error broadcasting message: {e}")

def broadcast_model(model):
    logger.info("Starting model broadcast to all clients")
    serialized = pickle.dumps(model)
    size = len(serialized)

    for client_socket in clients.keys():
        try:
            header = f"MODEL:{size}\n".encode('utf-8')
            client_socket.sendall(header)
            client_socket.sendall(serialized)
            logger.info(f"Sent model to client {clients[client_socket][1]}")
        except Exception as e:
            logger.error(f"Error sending model to client: {e}")

def send_models(client_socket):
    global broadcast_to_all_clients, null_model_sent
    
    while len(clients) < 2 and not should_exit:
        threading.Event().wait(2)
    
    if not broadcast_to_all_clients and not should_exit:
        try:
            logger.info("Starting initial model distribution")
            broadcast_message('Ok we Have 2 clients, We can start FL-Training, Sending Null Model')
            broadcast_model(model_null)
            broadcast_to_all_clients = True
            null_model_sent = True
            logger.info("Successfully sent null model to all clients")
            
        except Exception as e:
            logger.error(f"Error in sending model: {e}")
            del clients[client_socket]

def receive_models(client_socket, client_id):
    global is_training_complete, received_models, should_exit
    
    try:
        if client_id in received_models:
            logger.info(f"Already received model from client {client_id}")
            return
            
        size_data = client_socket.recv(4)
        if not size_data:
            return
        size = struct.unpack('!I', size_data)[0]
        
        logger.info(f"Expecting model of size {size} from client {client_id}")
        data = b''
        while len(data) < size:
            chunk = client_socket.recv(min(size - len(data), 4096))
            if not chunk:
                break
            data += chunk

        if len(data) == size:
            received_model = pickle.loads(data)
            models.append(received_model)
            received_models.add(client_id)
            
            logger.info(f"Successfully received model from client {client_id}")
            logger.info(f"Total models received: {len(models)}")
            logger.info(f"Models received from clients: {received_models}")
        
            client_socket.sendall("MODEL_RECEIVED\n".encode('utf-8'))
            
            if len(models) == 2:
                is_training_complete = True
                logger.info("Training complete! All models received")
                broadcast_message("Training Complete! Closing connections...")
                time.sleep(1)
                close_all_connections()
                
    except Exception as e:
        logger.error(f"Error receiving model from client {client_id}: {e}")

def receive():
    global should_exit
    client_counter = 0
    logger.info("Server is running and waiting for 2 clients...")
    
    server.settimeout(1.0)
    
    while not should_exit:
        try:
            client_socket, address = server.accept()
            client_counter += 1
            client_id = f"client_{client_counter}"
            clients[client_socket] = (address, client_id)
            
            logger.info(f"New connection established with {address} (ID: {client_id})")
            
            if len(clients) <= 2:
                broadcast_message(f'New User has joined the Training Session. Current clients: {len(clients)}')
                broadcast_message(f'Client {client_id} connected, Will be sending you the Training Algo')

                send_thread = threading.Thread(target=send_models, args=(client_socket,))
                send_thread.start()
                
                if len(clients) == 2:
                    logger.info("Both clients connected, starting model reception")
                    for client_sock, (addr, cid) in clients.items():
                        receive_thread = threading.Thread(target=receive_models, args=(client_sock, cid))
                        receive_thread.start()
            else:
                logger.warning(f"Rejected connection from {address} - server full")
                client_socket.sendall("Server is full. Try again later.\n".encode('utf-8'))
                client_socket.close()

        except socket.timeout:
            continue
        except Exception as e:
            if not should_exit:
                logger.error(f"Error in main loop: {e}")

if __name__ == "__main__":
    try:
        receive()
        while not should_exit:
            time.sleep(0.1)
        logger.info("Server shutdown complete")
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
        close_all_connections()
        sys.exit(0)