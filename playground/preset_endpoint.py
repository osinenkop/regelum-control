import rcognita as rc


@rc.main(config_path="general", config_name="main")
def launch(cfg):
    scenario = ~cfg.scenario
    if scenario.howanim in rc.ANIMATION_TYPES_REQUIRING_ANIMATOR:
        try:
            animator = ~cfg.animator
        except:
            raise NotImplementedError("Can't instantiate animator for your system")
        if scenario.howanim == "live":
            animator.play_live()
        elif scenario.howanim in rc.ANIMATION_TYPES_REQUIRING_SAVING_SCENARIO_PLAYBACK:
            scenario.run()
            animator.playback()
    else:
        scenario.run()

    return 0


if __name__ == "__main__":
    job_results = launch()
    pass
