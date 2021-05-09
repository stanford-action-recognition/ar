class Path(object):
    @staticmethod
    def db_dir(database):
        if database == "UCF101":
            root_dir = "./data/UCF-101"  # folder that contains class labels
            output_dir = "./data/UCF101"  # Save preprocess data into output_dir
            return root_dir, output_dir
        elif database == "HMDB51":
            root_dir = "./data/hmdb51_org"
            output_dir = "./data/HMDB51"
            return root_dir, output_dir
        else:
            print("Database {} not available.".format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return "./c3d-pretrained.pth"
