def __getattr__(name):
    if name == "Trm4Rec":
        from lib.Trm4Rec_trainer import Trm4Rec

        return Trm4Rec
    if name == "DINTrain":
        from lib.DIN_trainer import DINTrain

        return DINTrain
    if name == "DeepInterestNetwork":
        from lib.DIN_Model import DeepInterestNetwork

        return DeepInterestNetwork
    if name == "Train_instance":
        from lib.generate_training_batches import Train_instance

        return Train_instance
    raise AttributeError(name)
