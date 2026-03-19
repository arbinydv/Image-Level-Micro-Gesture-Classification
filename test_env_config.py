import mediapipe as mp
import torch
import platform

def test_config_env():
    print(" Platform ---> ", platform.machine())
    print("Pytorch version..", torch.__version__)
    print("GPU Available: ", torch.cuda.is_available())
    print("\n TESTING MEDIAPIPE INSTALLATION AND CONFIGURATION")
    print("Media pipe version is .....", mp.__version__)

    try:
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(static_image_mode=True)
        print("Media pipe of version 0.10.9 is installed that supports holistics")
        print(" Using Holistics is really nice as I don't have to download tasks")
        holistic.close()
    except Exception as e:
        print(f" Mediapipe error: {e}")

if __name__ == "__main__":
    test_config_env()

