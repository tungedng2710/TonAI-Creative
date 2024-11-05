from diffusers import DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, FlowMatchEulerDiscreteScheduler

schedulers = {"Euler": (FlowMatchEulerDiscreteScheduler,
                        {}),
              "DPM++ 2M": (DPMSolverMultistepScheduler,
                           {}),
              "DPM++ 2M Karras": (DPMSolverMultistepScheduler,
                                  {"use_karras_sigmas": True}),
              "DPM++ 2M SDE": (DPMSolverMultistepScheduler,
                               {"algorithm_type": "sde-dpmsolver++"},
                               ),
              "DPM++ 2M SDE Karras": (DPMSolverMultistepScheduler,
                                      {"use_karras_sigmas": True,
                                       "algorithm_type": "sde-dpmsolver++"},
                                      ),
              "DPM++ SDE": (DPMSolverSinglestepScheduler,
                            {}),
              "DPM++ SDE Karras": (DPMSolverSinglestepScheduler,
                                   {"use_karras_sigmas": True}),
              }


def apply_scheduler(name, pipeline):
    # Check if the scheduler name exists in the schedulers dictionary
    if name in schedulers:
        scheduler_class, config = schedulers[name]
        # Update the pipeline's scheduler with the selected scheduler class and
        # configuration
        pipeline.scheduler = scheduler_class.from_config(
            pipeline.scheduler.config, **config)
    return pipeline
