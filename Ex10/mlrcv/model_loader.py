import torch
from torchvision import models

import typing

def build_classifier(num_class: int) -> torch.nn.modules.container.Sequential:
  
    num_class =13
    
    classifier = torch.nn.Sequential(

        torch.nn.Linear(512, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear( 512, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(256, num_class),
        torch.nn.LogSoftmax(dim=1),

       

        
    #     torch.nn.Linear(1024, 512),
    #     torch.nn.ReLU(),
    #     torch.nn.Dropout(0.5),
    #     torch.nn.Linear(256, num_class),
    #     torch.nn.LogSoftmax(dim=1)
    )



# Example usage
    # input_features = torch.randn(1, 2048)  # Example input features from ResNet
    # output = build_classifier(input_features)  # Classify the input features
    # predicted_class = torch.argmax(output, dim=1)
    # print("Predicted class:", predicted_class.item())

    return classifier





# num_classes = 13
# classifier = build_classifier(num_classes)
"""
    This function builds the classifier to classify the resnet output features:

    Args:
        - num_class (int): number of class on the data to be defined as the classifier output features size

    Returns:
        - classifier (torch.nn.modules.container.Sequential): the classifier model to classify the resnet output features
    """