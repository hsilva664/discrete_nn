import torch
import numpy as np
import itertools

def xnor_distance(x1, x2):
    return (x1 * x2 + (1-x1) * (1-x2)).mean()

def safe_clip(x, eps=1e-8):
    return torch.clip(x, eps, 1.0)

def safe_log_prob(x, eps=1e-8):
    return torch.log(safe_clip(x, eps))

def safe_log_prob_1p(x, eps=1e-8):
    return torch.log1p(torch.clip(x, -(1 - eps), 0.0))

def v_from_u(u, logits):
    u_prime = torch.sigmoid(-logits)
    with torch.no_grad():
        v_1 = (u - u_prime) / safe_clip(1 - u_prime)
        v_1 = torch.clip(v_1, 0, 1)
        v_0 = u / safe_clip(u_prime)
        v_0 = torch.clip(v_0, 0, 1)
        
    v_1 = v_1 * (1 - u_prime) + u_prime
    v_0 = v_0 * u_prime

    v = torch.where(u > u_prime, v_1, v_0)
    v = v + (-v + u).detach()
    return v

def cosine_mask(x):
    return 0.5 * (1 - torch.cos(x))

def cosine_mask_inv(mask_p):
    # The function is not invertible, but we restrict it to the inverval [0,Pi]
    return -torch.arccos(2*mask_p-1) + np.pi

def expected_value(logits, function, mask_function=torch.sigmoid, bs=1000):
    total = 0.
    sig_logits = mask_function(logits)
    one_m_sig_logits = 1 - sig_logits
    d = sig_logits.shape[0]
    frame = 2. ** (torch.arange(d)).to(torch.get_default_dtype())
    examples_read = 0
    while examples_read < (2**d):
        new_examples_read = min(examples_read + bs, 2 ** d)
        x_dec = torch.arange(examples_read, new_examples_read)
        x_binary = x_dec.unsqueeze(-1).bitwise_and(frame.long()).ne(0).to(torch.get_default_dtype())
        y = function(x_binary)
        probs = torch.where(x_binary == 1., sig_logits, one_m_sig_logits )
        total += torch.sum(torch.prod(probs, dim=1) * y)
        examples_read = new_examples_read
    return total

def get_solution(num_latents, function, bs=1024):
    frame = 2. ** (torch.arange(num_latents)).to(torch.get_default_dtype())
    examples_read = 0
    solution = None
    solution_loss = torch.inf
    while examples_read < (2**num_latents):
        new_examples_read = min(examples_read + bs, 2**num_latents)
        x_dec = torch.arange(examples_read, new_examples_read)
        x_binary = x_dec.unsqueeze(-1).bitwise_and(frame.long()).ne(0).to(torch.get_default_dtype())
        y = function(x_binary)
        this_min, this_argmin = torch.min(y, 0)
        if this_min < solution_loss:
            solution_loss = this_min
            solution = x_binary[this_argmin]
        examples_read = new_examples_read
    return solution, solution_loss

def get_whole_table(num_latents, function, bs=1024):
    frame = 2. ** (torch.arange(num_latents)).to(torch.get_default_dtype())
    examples_read = 0
    table = torch.zeros(2**num_latents)
    while examples_read < (2**num_latents):
        new_examples_read = min(examples_read + bs, 2**num_latents)
        x_dec = torch.arange(examples_read, new_examples_read)
        x_binary = x_dec.unsqueeze(-1).bitwise_and(frame.long()).ne(0).to(torch.get_default_dtype())
        y = function(x_binary)
        table[examples_read:new_examples_read] = y
        examples_read = new_examples_read
    return table

def jacobian(y, x, create_graph=False):                                                               
    jac = []                                                                                          
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)):                                                                      
        grad_y[i] = 1.                                                                                
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))                                                           
        grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape)                                                
                                                                                                      
def hessian(y, x):                                                                                    
    return jacobian(jacobian(y, x, create_graph=True), x)      

class QFunc(torch.nn.Module):
    '''Control variate for RELAX'''

    def __init__(self, num_latents, hidden_size=10, nlayers=1):
        super().__init__()        
        l_list = [torch.nn.Sequential(torch.nn.Linear(num_latents if i == 0 else hidden_size, hidden_size), torch.nn.Tanh()) for i in range(nlayers)] + [torch.nn.Linear(hidden_size, 1)]
        self.all_l = torch.nn.Sequential(*l_list)

    def forward(self, z):
        # the multiplication by 2 and subtraction is from toy.py...
        # it doesn't change the bias of the estimator, I guess
        z = z * 2. - 1.
        return self.all_l(z)