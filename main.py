from train import Trainer
from inference_question import BeamSearcher
import os
import config_file
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
if __name__ == '__main__':



    if (config_file.Train_or_not):
        print("train start...")

        Train = Trainer()
        Train.train()
        os.system('shutdown -s -t 5')
    else:

        print("search")
        BeamSearcher = BeamSearcher('save_model_new/1.191978epoch:21', 'out/inferenceL22221.txt')

        BeamSearcher.decode()
        # inference = Inference()
        # inference.inference()
