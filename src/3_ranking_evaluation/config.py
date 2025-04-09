import os


basepath = "/media/Workspace/Datasets"
top_k = 10
model_names = ["ResNet", "AttentionNet", "HRNet", "ResNeSt", "MobileFaceNet", "RepVGG"]




# each path should point out to the previously extracted embeddings
models_data_HR_HR = {
    "DiveFace": [
        os.path.join(basepath, "DiveFace/TestDatasetResNet.pt.npy"),  # ResNet
        os.path.join(basepath, "DiveFace/embeddings/DatasetAttentionNet+NPCFace.pt.npy"),  # AttentionNet
        os.path.join(basepath, "DiveFace/embeddings/DatasetHRNet+NPCFace.pt.npy"),  # HRNet
        os.path.join(basepath, "DiveFace/embeddings/DatasetResNeSt+NPCFace.pt.npy"),  # ResNeSt
        os.path.join(basepath, "DiveFace/embeddings/DatasetMobileFaceNetNPCFaceOnHR.pt.npy"), # MobileFaceNet
        os.path.join(basepath, "DiveFace/embeddings/DatasetRepVGG+NPCFace.pt.npy"),  # RepVGG
    ],
    "VGGFace2": [
        os.path.join(basepath, "VGG-Face2/data/vggface2_test/croppedResNet152+NPCFace.pt.npy"),  # ResNet
        os.path.join(basepath, "VGG-Face2/data/vggface2_test/croppedAttentionNet+NPCFace.pt.npy"), # AttentionNet
        os.path.join(basepath, "VGG-Face2/data/vggface2_test/croppedHRNet+NPCFace.pt.npy"),  # HRNet
        os.path.join(basepath, "VGG-Face2/data/vggface2_test/croppedResNeSt+NPCFace.pt.npy"),  # ResNeSt
        os.path.join(basepath, "VGG-Face2/data/vggface2_test/croppedMobileFaceNetNPCFaceOnHR.pt.npy"), # MobileFaceNet
        os.path.join(basepath, "VGG-Face2/data/vggface2_test/croppedRepVGG+NPCFace.pt.npy"),  # RepVGG
    ],
    "CelebA": [
        os.path.join(basepath, "CelebA/Aligned_DatasetResNet152+NPCFace.pt.npy"),  # ResNet
        os.path.join(basepath, "CelebA/Aligned_DatasetAttentionNet+NPCFace.pt.npy"),  # AttentionNet
        os.path.join(basepath, "CelebA/Aligned_DatasetHRNet+NPCFace.pt.npy"),  # HRNet
        os.path.join(basepath, "CelebA/Aligned_DatasetResNeSt+NPCFace.pt.npy"),  # ResNeSt
        os.path.join(basepath, "CelebA/Aligned_DatasetMobileFaceNetNPCFaceOnHR.pt.npy"),
        # MobileFaceNet
        os.path.join(basepath, "CelebA/Aligned_DatasetRepVGG+NPCFace.pt.npy"),  # RepVGG
    ],
    "RFW": [
        os.path.join(basepath, "RFW/DatasetResNet152+NPCFace.pt.npy"),  # ResNet
        os.path.join(basepath, "RFW/DatasetAttentionNet+NPCFace.pt.npy"),  # AttentionNet
        os.path.join(basepath, "RFW/DatasetHRNet+NPCFace.pt.npy"),  # HRNet
        os.path.join(basepath, "RFW/DatasetResNeSt+NPCFace.pt.npy"),  # ResNeSt
        os.path.join(basepath, "RFW/DatasetMobileFaceNetNPCFaceOnHR.pt.npy"),  # MobileFaceNet
        os.path.join(basepath, "RFW/DatasetRepVGG+NPCFace.pt.npy"),  # RepVGG
    ],
    "BUPT": [
        os.path.join(basepath, "BUPT/Equalizedface/old/DatasetHRResNet152+NPCFace.pt.npy"),  # ResNet
        os.path.join(basepath, "BUPT/Equalizedface/old/DatasetHRAttentionNet+NPCFace.pt.npy"),  # AttentionNet
        os.path.join(basepath, "BUPT/Equalizedface/old/DatasetHRHRNet+NPCFace.pt.npy"),  # HRNet
        os.path.join(basepath, "BUPT/Equalizedface/old/DatasetHRResNeSt+NPCFace.pt.npy"),  # ResNeSt
        os.path.join(basepath, "BUPT/DatasetHRMobileFaceNetNPCFaceOnHR.pt.npy"),  # MobileFaceNet
        os.path.join(basepath, "BUPT/Equalizedface/old/DatasetHRRepVGG+NPCFace.pt.npy"),  # RepVGG
    ],
}

# LR model on LR images
models_data_LR_LR = {
    "DiveFace": [
        os.path.join(basepath, "DiveFace/embeddings/WildDatasetResNet152LR+NPCFace.pt.npy"),  # ResNet
        os.path.join(basepath, "DiveFace/embeddings/WildDatasetAttentionNet+NPCFaceLR.pt.npy"), # AttentionNet
        os.path.join(basepath, "DiveFace/embeddings/WildDatasetHRNet+NPCFaceLR.pt.npy"),  # HRNet
        os.path.join(basepath, "DiveFace/embeddings/WildDatasetResNeSt+NPCFaceLR.pt.npy"),  # ResNeSt
        os.path.join(basepath, "DiveFace/embeddings/WildDatasetMobileFaceNetNPCFaceOnLR.pt.npy"), # MobileFaceNet
        os.path.join(basepath, "DiveFace/embeddings/WildDatasetRepVGG+NPCFaceLR.pt.npy"),  # RepVGG
    ],
    "VGGFace2": [
        os.path.join(basepath, "VGG-Face2/data/vggface2_test/WildcroppedResNet152LR+NPCFace.pt.npy"),  # ResNet
        os.path.join(basepath, "VGG-Face2/data/vggface2_test/WildcroppedAttentionNet+NPCFaceLR.pt.npy"), # AttentionNet
        os.path.join(basepath, "VGG-Face2/data/vggface2_test/WildcroppedHRNet+NPCFaceLR.pt.npy"),  # HRNet
        os.path.join(basepath, "VGG-Face2/data/vggface2_test/WildcroppedResNeSt+NPCFaceLR.pt.npy"),  # ResNeSt
        os.path.join(basepath, "VGG-Face2/data/vggface2_test/WildcroppedMobileFaceNetNPCFaceOnLR.pt.npy"), # MobileFaceNet
        os.path.join(basepath, "VGG-Face2/data/vggface2_test/WildcroppedRepVGG+NPCFaceLR.pt.npy"),  # RepVGG
    ],
    "CelebA": [
        os.path.join(basepath, "CelebA/WildAligned_DatasetResNet152LR+NPCFace.pt.npy"),  # ResNet
        os.path.join(basepath, "CelebA/WildAligned_DatasetAttentionNet+NPCFaceLR.pt.npy"),  # AttentionNet
        os.path.join(basepath, "CelebA/WildAligned_DatasetHRNet+NPCFaceLR.pt.npy"),  # HRNet
        os.path.join(basepath, "CelebA/WildAligned_DatasetResNeSt+NPCFaceLR.pt.npy"),  # ResNeSt
        os.path.join(basepath, "CelebA/WildAligned_DatasetMobileFaceNetNPCFaceOnLR.pt.npy"),
        # MobileFaceNet
        os.path.join(basepath, "CelebA/WildAligned_DatasetRepVGG+NPCFaceLR.pt.npy"),  # RepVGG
    ],
    "RFW": [
        os.path.join(basepath, "RFW/DatasetLRResNet152LR+NPCFace.pt.npy"),  # ResNet
        os.path.join(basepath, "RFW/DatasetLRAttentionNet+NPCFaceLR.pt.npy"),  # AttentionNet
        os.path.join(basepath, "RFW/DatasetLRHRNet+NPCFaceLR.pt.npy"),  # HRNet
        os.path.join(basepath, "RFW/DatasetLRResNeSt+NPCFaceLR.pt.npy"),  # ResNeSt
        os.path.join(basepath, "RFW/DatasetLRMobileFaceNetNPCFaceOnLR.pt.npy"),  # MobileFaceNet
        os.path.join(basepath, "RFW/DatasetLRRepVGG+NPCFaceLR.pt.npy"),  # RepVGG
    ],
    "BUPT": [
        os.path.join(basepath, "BUPT/Equalizedface/old/DatasetLRResNet152LR+NPCFace.pt.npy"),  # ResNet
        os.path.join(basepath, "BUPT/Equalizedface/old/DatasetLRAttentionNet+NPCFaceLR.pt.npy"),  # AttentionNet
        os.path.join(basepath, "BUPT/Equalizedface/old/DatasetLRHRNet+NPCFaceLR.pt.npy"),  # HRNet
        os.path.join(basepath, "BUPT/Equalizedface/old/DatasetLRResNeSt+NPCFaceLR.pt.npy"),  # ResNeSt
        os.path.join(basepath, "BUPT/DatasetLRMobileFaceNetNPCFaceOnLR.pt.npy"),  # MobileFaceNet
        os.path.join(basepath, "BUPT/Equalizedface/old/DatasetLRRepVGG+NPCFaceLR.pt.npy"),  # RepVGG
    ],
}


# HR points to the root folder of the High resolution dataset
# LR points to the root folder of the Low resolution (degraded) dataset
path_conv_dict = {
    "DiveFace": {"HR": "/Dataset/", "LR": "/WildDataset/"},
    "VGGFace2": {"HR": "/cropped/", "LR": "/Wildcropped/"},
    "CelebA": {"HR": "/Aligned_Dataset/", "LR": "/WildAligned_Dataset/"},
    "RFW": {"HR": "/Dataset/", "LR": "/DatasetLR/"},
    "BUPT": {"HR": "/DatasetHR/", "LR": "/DatasetLR/"},
}

info = {
    "DiveFace" : {
        "test_set_file" : os.path.join(basepath, "DiveFace/testHR.txt"),
        "json": os.path.join(basepath, "DiveFace/DiveFace_test_HR.json")
    },
    "VGGFace2" : {
        "test_set_file" : os.path.join(basepath, "VGG-Face2/data/vggface2_test/testHR.txt"),
        "json": os.path.join(basepath, "VGG-Face2/data/vggface2_test/test_HR.json")
    },
    "CelebA" : {
        "test_set_file" : os.path.join(basepath, "CelebA/testHR.txt"),
        "json": os.path.join(basepath, "CelebA/test_HR.json")
    },
    "RFW" : {
        "test_set_file" : os.path.join(basepath, "RFW/testHR.txt"),
        "json": os.path.join(basepath, "RFW/test_HR.json")
    },
    "BUPT" : {
        "test_set_file" : os.path.join(basepath, "BUPT/testHR.txt"),
        "json": os.path.join(basepath, "BUPT/test_HR.json")
    },
}


