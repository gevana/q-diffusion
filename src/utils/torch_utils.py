
def add_full_name_to_module(model) -> None:
    # add full name to each module
    for name, module in model.named_modules():
        module.full_name = name


def save_input_hook(module, input, output):
    if not hasattr(module, "saved_inputs"):
        module.saved_inputs = []  # Create attribute if not exists
    #print(f"Saving input # {len(module.saved_inputs)}")
    module.saved_inputs.append(input)  # A


def register_forword_hook(module, hook):
    handle = module.register_forward_hook(hook)
    return handle
