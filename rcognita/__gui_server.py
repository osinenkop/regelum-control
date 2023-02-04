import os, sys, shelve, time
import signal


def get_reports():
    return sorted(list(filter(lambda name: name.startswith('report'), [os.fsencode(file).decode('utf-8').split('.')[0] for file in os.listdir(sys.argv[1])])))


if __name__ == "__main__":
    try:
        import streamlit as st
        import pandas as pd


        containers = {}
        os.chdir(sys.argv[1])
        while True:
            time.sleep(1)
            reports = get_reports()
            for report in reports:
                if report not in containers:
                    st.title(report.split("_")[1])
                    containers[report] = {"body" : st.container()}
                    with shelve.open(report) as r:
                        with containers[report]["body"]:
                            st.components.v1.html(r["overrides_html"])
                            st.components.v1.html(f'<a href="file://{r["path"] + "/SUMMARY.html"}">View Summary</a>')
                            containers[report]["scenario_caption"] = st.empty()
                            containers[report]["scenario_caption"].write("Scenario:")
                            containers[report]["scenario_progress"] = st.progress(0.0)
                            st.write("Current episode:")
                            containers[report]["episode_progress"] = st.progress(0.0)
                            pid = r["pid"]
                            containers[report]["button_block"] = st.empty()
                            def stop():
                                with shelve.open(report) as r_:
                                    r_["terminate"] = True

                            containers[report]["button_clicked"] = containers[report]["button_block"].button(label="Stop", on_click=stop, key=f'button_{report.split("_")[1]}')
                if containers[report]["button_clicked"]:
                    containers[report]["button_block"].empty()
                    containers[report]["button_clicked"] = False
                    with containers[report]["body"]:
                        st.error("Process stopped by user.")
                with shelve.open(report) as r:
                    if "traceback" in r:
                        containers[report]["button_block"].empty()
                        with containers[report]["body"]:
                            st.error(r["traceback"])
                        del r["traceback"]
                    containers[report]["episode_progress"].progress(r["elapsed_relative"] if "elapsed_relative" in r else 0.0)
                    if "episode_current" in r:
                        containers[report]["scenario_progress"].progress(r["episode_current"] / r["episode_total"])
                        containers[report]["scenario_caption"].write(f'Scenario ({r["episode_current"]}/{r["episode_total"]}):')
    except (RuntimeError, FileNotFoundError):
        st.stop()

