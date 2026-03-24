import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

log_path = os.path.join(os.getcwd(), "logs")
os.makedirs(log_path, exist_ok=True)

LOG_FILEPATH = os.path.join(log_path, LOG_FILE)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILEPATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    filemode='w'  # 'w' for write mode (overwrites), 'a' for append mode
)

# Create a logger instance
logger = logging.getLogger(__name__)