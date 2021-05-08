class Path(object):
    @staticmethod
    def db_dir(database):
        if database == "ucf101":
            root_dir = "./data/UCF-101"  # folder that contains class labels
            output_dir = "./data/ucf101"  # Save preprocess data into output_dir
            return root_dir, output_dir
        elif database == "hmdb51":
            root_dir = "./data/hmdb51_org"
            output_dir = "./data/hmdb51"
            return root_dir, output_dir
        else:
            print("Database {} not available.".format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return "./c3d-pretrained.pth"
