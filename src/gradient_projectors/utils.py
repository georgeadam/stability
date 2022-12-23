import torch
from functorch import grad, make_functional, make_functional_with_buffers, vmap


def parameters_to_grad_vector(parameters):
    # Flag for the device where the parameter is located
    param_device = None

    vec = []

    for param in parameters:
        # Ensure the parameters are located in the same device

        vec.append(param.grad.view(-1))

    return torch.cat(vec)


def grad_vector_to_parameters(vec, parameters):
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'.format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0

    for param in parameters:
        # Ensure the parameters are located in the same device

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        # param.data = vec[pointer:pointer + num_param].view_as(param).data
        param.grad = vec[pointer:pointer + num_param].view_as(param).clone()

        # Increment the pointer
        pointer += num_param


def get_per_sample_gradients(model, x, y, loss_fn):
    func_model, params = make_functional(model)
    per_sample_gradients = vmap(grad(loss_wrapper(func_model)), in_dims=(None, 0, 0, None))(params, x, y, loss_fn)
    flattened_grads = [temp_grad.view(temp_grad.shape[0], -1) for temp_grad in per_sample_gradients]
    flattened_grads = torch.concat(flattened_grads, dim=1)

    return flattened_grads


def loss_wrapper(func_model):
    def compute_loss(params, x, y, loss_fn):
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        out = func_model(params, x)

        return loss_fn(out, y)

    return compute_loss


def get_per_sample_gradients_buffer(model, x, y, loss_fn):
    func_model, params, buffers = make_functional_with_buffers(model)
    per_sample_gradients = vmap(grad(loss_wrapper_buffer(func_model)), in_dims=(None, None, 0, 0, None))(params, buffers, x, y, loss_fn)
    flattened_grads = [temp_grad.view(temp_grad.shape[0], -1) for temp_grad in per_sample_gradients]
    flattened_grads = torch.concat(flattened_grads, dim=1)

    return flattened_grads


def loss_wrapper_buffer(func_model):
    def compute_loss(params, x, y, loss_fn):
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        out = func_model(params, x)

        return loss_fn(out, y)

    return compute_loss