import os, sys, shelve, time
import signal

import dill
import matplotlib.pyplot as plt

import rcognita


def get_reports():
    return sorted(list(set(filter(lambda name: name.startswith('report'), [os.fsencode(file).decode('utf-8').split('.')[0] for file in os.listdir(sys.argv[1])]))))

class Stopper:
    def __init__(self, report_):
        self.report = report_

    def __call__(self):
        with shelve.open(self.report) as r_:
            r_["terminate"] = True

if __name__ == "__main__":
    try:
        import streamlit as st
        import pandas as pd


        containers = {}
        os.chdir(sys.argv[1])
        counter = 0
        while True:
            time.sleep(1)
            counter += 1
            reports = get_reports()
            for num_report, report in enumerate(reports):
                if report not in containers:
                    st.title(report.split("_")[1])
                    containers[report] = {"body" : st.container()}
                    with shelve.open(report) as r:
                        with containers[report]["body"]:
                            st.components.v1.html(r["overrides_html"])
                            st.write("Summary:")
                            containers[report]["path"] = r["path"]
                            st.code(f'file://{containers[report]["path"] + "/SUMMARY.html"}')
                            containers[report]["scenario_caption"] = st.empty()
                            containers[report]["scenario_caption"].write("Scenario:")
                            containers[report]["scenario_progress"] = st.progress(0.0)
                            st.write("Current episode:")
                            containers[report]["episode_progress"] = st.progress(0.0)
                            pid = r["pid"]
                            containers[report]["plots_container"] = st.expander(f"Plots")
                            with containers[report]["plots_container"]:
                                containers[report]["plots"] = st.empty()
                            containers[report]["button_block"] = st.empty()
                            containers[report]["episode"] = 0
                            containers[report]["button_clicked"] = containers[report]["button_block"].button(label="Stop", on_click=Stopper(report), key=f'button_{report}')
                if containers[report]["button_clicked"]:
                    containers[report]["button_block"].empty()
                    containers[report]["button_clicked"] = False
                    with containers[report]["body"]:
                        st.error("Process stopped by user.")
                load_callbacks = False
                with shelve.open(report) as r:
                    if "episode_current" in r:
                        current_episode = r["episode_current"]
                        if current_episode > containers[report]["episode"]:
                            containers[report]["episode"] = current_episode
                            load_callbacks = True
                if load_callbacks:
                    with open(containers[report]["path"] + "/callbacks.dill", "rb") as f:
                        containers[report]["callbacks"] = dill.load(f)
                    with containers[report]["plots"].container():
                        for callback in containers[report]["callbacks"]:
                            if isinstance(callback, rcognita.HistoricalCallback):
                                fig = callback.plot_gui()
                                if fig is not None:
                                    st.pyplot(fig)
                with shelve.open(report) as r:
                    if "traceback" in r:
                        containers[report]["button_block"].empty()
                        with containers[report]["body"]:
                            st.error(r["traceback"])
                        del r["traceback"]
                    containers[report]["episode_progress"].progress(r["elapsed_relative"] if "elapsed_relative" in r else 0.0)
                    if "episode_total" in r:
                        containers[report]["scenario_progress"].progress(r["episode_current"] / r["episode_total"])
                        containers[report]["scenario_caption"].write(f'Scenario ({r["episode_current"]}/{r["episode_total"]}):')
    except (RuntimeError, FileNotFoundError):
        st.stop()

