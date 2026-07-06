import os
import sys
from pathlib import Path
import socket
from sys import platform


#  Параметры запуска.
docker_run = False
port = 7627

if platform == 'linux':
    ip = "192.168.19.123"
else:
    ip = "192.168.18.194"


def get_host_ip():
    """Функция получения ip адреса машины хоста."""
    try:
        ip = socket.gethostbyname(socket.gethostname() + ".local")
    except:
        ip = socket.gethostbyname(socket.gethostname())
    return ip


#  Настройки сети.
if not docker_run:
    worker_host = "127.0.0.1"
    main_host = ip
else:
    ip = get_host_ip() # default: "91.217.196.199"
    worker_host = "msinv_main_worker"
    main_host = "msinv_main_server"


#  Структура папок.
ROOT_DIR = os.getcwd() #path.abspath(os.path.join(__file__, "."))
temp_folder = os.path.join(ROOT_DIR, "temp")
tasks_folder = os.path.join(ROOT_DIR, "tasks")
results_folder = os.path.join(ROOT_DIR, "results")
logs_folder = os.path.join(ROOT_DIR, "logs")
download_folder = os.path.join(ROOT_DIR, "download")
thread_limit = 2

#  Обслуживание модулей.
MODULES = {"ilab_invoice_handler":"modules.ilab_invoice_handler", "ilab_cact_handler":"modules.ilab_cact_handler"}

modules_routing = {
    "invoice":"ilab_invoice_handler",
    "plist":"",
    "cact":"ilab_cact_handler"
}
