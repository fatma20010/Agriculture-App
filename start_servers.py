import subprocess
import os
import sys
import time
import signal
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Keep track of all processes we start
processes = []

def start_process(cmd, name):
    """Start a process and return it"""
    logger.info(f"Starting {name}...")
    if os.name == 'nt':  # Windows
        process = subprocess.Popen(cmd, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:  # Unix/Linux/Mac
        process = subprocess.Popen(cmd, shell=True)
    processes.append((process, name))
    logger.info(f"{name} started with PID {process.pid}")
    return process

def cleanup(signum=None, frame=None):
    """Clean up all started processes"""
    logger.info("Shutting down all servers...")
    for process, name in processes:
        try:
            logger.info(f"Terminating {name} (PID {process.pid})...")
            if os.name == 'nt':  # Windows
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)])
            else:
                process.terminate()
                process.wait(timeout=5)
        except Exception as e:
            logger.error(f"Error stopping {name}: {e}")
    logger.info("All servers stopped")
    sys.exit(0)

# Register signal handlers for clean shutdown
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

if __name__ == "__main__":
    try:
        # Start the veg.py server (Farm analysis)
        veg_process = start_process("python veg.py", "Veg Server")
        time.sleep(2)  # Wait for server to start
        
        # Start the grass.py server (Hotel/Stadium analysis)
        grass_process = start_process("python grass.py", "Grass Server")
        time.sleep(2)  # Wait for server to start
        
        # Start the proxy server
        proxy_process = start_process("python proxy_server.py", "Proxy Server")
        time.sleep(2)  # Wait for server to start
        
        logger.info("All servers started successfully!")
        logger.info("Press Ctrl+C to stop all servers")
        
        # Keep the main process running
        while True:
            time.sleep(1)
            # Check if any process has terminated unexpectedly
            for i, (process, name) in enumerate(processes):
                if process.poll() is not None:
                    exit_code = process.returncode
                    logger.error(f"{name} terminated unexpectedly with exit code {exit_code}")
                    cleanup()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        cleanup()
    except Exception as e:
        logger.error(f"Error: {e}")
        cleanup() 