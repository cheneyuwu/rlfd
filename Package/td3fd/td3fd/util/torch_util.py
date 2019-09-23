import torch


def check_module(module):
    print("\ncheck modules\n============================")
    for i in module.named_modules():
        print(i)
    print("\ncheck children\n============================")
    for i in module.named_children():
        print(i)
    print("\ncheck parameters\n============================")
    for i in module.named_parameters():
        print(i)
    print("\ncheck buffers\n============================")
    for i in module.named_buffers():
        print(i)
    print("\ncheck module state dict\n============================")
    for i, v in module.state_dict().items():
        print(i, v)
