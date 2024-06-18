from diffusers import DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler

schedulers = {
    "DPMPP_2M": (DPMSolverMultistepScheduler, {}),
    "DPMPP_2M_Karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
    "DPMPP_2M_SDE": (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++"}),
    "DPMPP_2M_SDE_Karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True, "algorithm_type": "sde-dpmsolver++"}),
    "DPMPP_SDE": (DPMSolverSinglestepScheduler, {}),
    "DPMPP_SDE_Karras": (DPMSolverSinglestepScheduler, {"use_karras_sigmas": True}),
}
