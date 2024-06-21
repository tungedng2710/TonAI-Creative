from diffusers import DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, FlowMatchEulerDiscreteScheduler

schedulers = {
    "Euler": (FlowMatchEulerDiscreteScheduler, {}),
    "DPMPP_2M": (DPMSolverMultistepScheduler, {}),
    "DPMPP_2M_Karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
    "DPMPP_2M_SDE": (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++"}),
    "DPMPP_2M_SDE_Karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True, "algorithm_type": "sde-dpmsolver++"}),
    "DPMPP_SDE": (DPMSolverSinglestepScheduler, {}),
    "DPMPP_SDE_Karras": (DPMSolverSinglestepScheduler, {"use_karras_sigmas": True}),
}

def get_scheduler(name, pipeline):
    # Check if the scheduler name exists in the schedulers dictionary
    if name in schedulers:
        scheduler_class, config = schedulers[name]
        # Update the pipeline's scheduler with the selected scheduler class and configuration
        pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config, **config)
    else:
        raise ValueError(f"Scheduler '{name}' not found in the available schedulers.")