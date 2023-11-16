import regelum as rg


@rg.main(config_path="stable-presets", config_name="main")
def launch(cfg):
    pipeline = ~cfg.pipeline
    pipeline.run()


if __name__ == "__main__":
    job_results = launch()
    pass
