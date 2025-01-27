
def add_full_name_to_module(model) -> None:
    # add full name to each module
    for name, module in model.named_modules():
        module.full_name = name

