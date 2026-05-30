from src.algorithms.dual_qd import dual_qd_optimizer
from src.algorithms.smooth_qd import SmoothSchedulerWrapper

def smooth_dual_qd_optimizer(solution_dim, metrics, params, gate_set):
    # 1. Instantiate base Dual optimizer
    # This gives us the DualChannelSchedulerWrapper and the archives
    scheduler, opt_arch, res_arch = dual_qd_optimizer(solution_dim, metrics, params, gate_set)
    
    # 2. Extract annealing params from QDParams (or defaults matching smooth_qd)
    tau0 = getattr(params, "tau0", 0.75)
    tau_min = getattr(params, "tau_min", 0.10)
    tau_decay = getattr(params, "tau_decay", 0.99)
    
    # 3. Wrap the Dual Scheduler in the Smooth Scheduler
    # SmoothSchedulerWrapper intercepts tell() to anneal tau, 
    # and forwards ask() to the Dual scheduler.
    wrapped_scheduler = SmoothSchedulerWrapper(scheduler, tau0, tau_min, tau_decay)
    
    return wrapped_scheduler, opt_arch, res_arch
