"""
This python script is used as a universal means of launching experiments. Systems, controllers and other
parameters are meant to be set via hydra's override syntax.
"""

import rcognita as rc


@rc.main(config_path="general", config_name="main")
def launch(cfg):
    """
    General launch script for an arbitrary configuration specified via command line arguments.
    """

    scenario = ~cfg.scenario
    if scenario.howanim in rc.ANIMATION_TYPES_REQUIRING_ANIMATOR:
        try:
            animator = ~cfg.animator
        except Exception as exc:
            raise NotImplementedError("Can't instantiate animator for your system") from exc
        if scenario.howanim == "live":
            animator.play_live()
        elif scenario.howanim in rc.ANIMATION_TYPES_REQUIRING_SAVING_SCENARIO_PLAYBACK:
            scenario.run()
            animator.playback()
    else:
        scenario.run()

    return 0


if __name__ == "__main__":
    JOB_RESULTS = launch()
    print(JOB_RESULTS["result"])
