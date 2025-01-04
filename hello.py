from tensorflow import config
from tensorflow.python.client import device_lib

def main():
    hl = "="*30
    print("Hello from tensortest!")
    print(hl)
    print(config.list_physical_devices('CPU'))
    print(hl)
    print(config.list_physical_devices('GPU'))
    print(hl)
    print(device_lib.list_local_devices())


if __name__ == "__main__":
    main()
