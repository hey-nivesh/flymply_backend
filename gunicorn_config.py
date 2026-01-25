import os

# Gunicorn configuration settings
bind = "0.0.0.0:" + str(os.getenv("PORT", 5000))
workers = 2  # Adjust based on available CPU cores. Since model is heavy, 2 is conservative.
threads = 1
timeout = 120  # increased timeout for model loading/processing
keepalive = 2
ws = "sync" # 'gevent' or 'eventlet' might be better for async but requires installation
loglevel = "info"
accesslog = "-"  # stdout
errorlog = "-"   # stderr

# Preload app to save memory?
# preloading might cause issues with PyTorch CUDA initialization in forks.
# Safe to leave False for PyTorch.
preload_app = False
