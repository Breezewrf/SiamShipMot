import shutil
import os


if __name__ == '__main__':     
    path = r"/home/shared/wrf/siam-mot/smd_dataset/raw_data"
    to_path = r""
    train_name = []
    test_name = []
    import pdb
    # pdb.set_trace()
    for _, f, _, in os.walk(path+"/train"):
        train_name.append(f)
        print("train video dir name {} is found\n".format(f))
        import pdb
        # pdb.set_trace()
    for _, f, _, in os.walk(path+"/test"):
        test_name.append(f)
        print("test video dir name {} is found\n".format(f))


    for p, f, v in os.walk(path):
        for video in v:
            name = video.split('.')[0]
            # pdb.set_trace()
            if name in train_name[0]:
                # pdb.set_trace()
                shutil.move(os.path.join(path, video), os.path.join(path, "train/" + name))
                print("video {} transmitted successfully".format(name))
            elif name in test_name[0]:
                # pdb.set_trace()
                shutil.move(os.path.join(path, video), os.path.join(path, "test/" + name))
                print("video {} transmitted successfully".format(name))

    print("done")         
    
