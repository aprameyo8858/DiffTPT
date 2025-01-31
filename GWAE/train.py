import os
import argparse

import vaetc
import yaml
import time
import models
from torchvision import datasets  

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("settings_path", type=str, help="path to settings YAML")
    parser.add_argument("--proceed", "-p", action="store_true", help="continue, load existing weights")
    parser.add_argument("--no_training", "-n", action="store_true", help="only evaluation (using with -p)")
    parser.add_argument("--seed", "-s", type=int, default=None, help="deterministic if seed is specified")
    args = parser.parse_args()
    #checkpoint.options['epochs'] = 20  # or whatever value is appropriate

    if args.seed is not None:
        vaetc.deterministic(args.seed)

    with open(args.settings_path, "r") as fp:
        options = yaml.safe_load(fp)
        options["hyperparameters"] = yaml.safe_dump(options["hyperparameters"])

    if not args.proceed:
        
        #root_path = '/Documents/AnyDesk/gwae/data/'
        #options['dataset'] = datasets.CelebA(root=root_path, download=False)

        checkpoint = vaetc.Checkpoint(options)
        #num_epochs = 20
        #checkpoint.options['epochs']= num_epochs
        if not args.no_training:
            start_time = time.time()
            vaetc.fit(checkpoint)
            end_time = time.time()
            total_time = end_time - start_time
            #print("The total time for the training for: ", num_epochs, " epochs is ", total_time)  

    else:

        checkpoint = vaetc.load_checkpoint(os.path.join(options["logger_path"], "checkpoint_last.pth"))
        if not args.no_training:
            vaetc.proceed(checkpoint, extend_epochs=None)

    vaetc.evaluate(checkpoint, checkpoint)
    print("The total time is ",total_time)
