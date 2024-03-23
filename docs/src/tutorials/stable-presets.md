Access and experiment with our pre-configurations with the following steps:

1. **Explore the Presets**: Visit our Github repository at [Regelum - Preconfigured Presets](https://github.com/osinenkop/regelum-control/tree/master/presets) to discover the range of available presets tailored for various agents and systems.

2. **Download and Run**: Seamlessly download the desired preset folder to your local machine. To initiate an experiment, navigate to the appropriate directory and execute a predefined bash script in the format:

    ```shell
    bash bash/{AGENT}/{SYSTEM}
    ```

   Replace {AGENT} with the name of the reinforcement learning agent, and {SYSTEM} with the target system you wish to simulate.

3. After running simulations, results are stored in the `regelum_data` folder. Launch MLflow UI using the following command in your terminal:

    ```shell
    cd regelum_data
    mlflow ui
    ```   

   The MLflow UI will present a overview of all experiments, allowing you to examine the metrics and outcomes conveniently.
