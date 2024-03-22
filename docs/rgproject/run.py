import regelum as rg


@rg.main(config_path="presets", config_name="main")
def launch(cfg):
    scenario = ~cfg.scenario  # instantiate the scenario from config
    scenario.run()  # run it


if __name__ == "__main__":
    job_results = launch()
    pass
