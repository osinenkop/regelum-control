# from __future__ import annotations
# from typing import Any, Dict, List, Optional, Callable
# from dataclasses import dataclass, field


# @dataclass(frozen=True)
# class Task:
#     @dataclass(frozen=True)
#     class Result:
#         data: Dict[str, Any] = field(default_factory=dict)

#         def __add__(self, other: Task.Result):
#             return self.data | other.data

#     @dataclass(frozen=True)
#     class Operands:
#         data: List[str] = field(default_factory=list)

#     name: str
#     callable_target: Callable = lambda _: None
#     returns: List = field(default_factory=list)
#     acquires_from: Dict[Task, List[str]] = field(default_factory=dict)

#     def __post_init__(self):
#         awaited_parameters = self.acquires_from.values()
#         if any(awaited_parameters):
#             awaited_parameters = list(set(sum(list(awaited_parameters), [])))
#             object.__setattr__(self, "operands", self.Operands(awaited_parameters))

#     def acquire(self, acquire_from: "Task", fields: List[str]) -> "Task":

#         if len(fields) > 0:
#             assert all(
#                 [field in acquire_from.returns for field in fields]
#             ), "acquire fields must be in returns of acquire_from task"
#         return Task(
#             callable_target=self.callable_target,
#             name=self.name,
#             returns=fields,
#             acquires_from={acquire_from: fields, acquire_from: acquire_from.returns},
#         )

#     def trace_leaf_tasks(self, leaf_tasks: List[Task]):
#         if self.acquires_from == {}:
#             leaf_tasks.append(self)
#         else:
#             for task in self.acquires_from:
#                 task.trace_leaf_tasks(leaf_tasks)
#         return leaf_tasks

#     def execute(self, other_tasks_result: Task.Result):
#         vars_to_substitute = {
#             k: v
#             for k, v in other_tasks_result.data.items()
#             if k in list(set(sum(list(self.acquires_from.values()), [])))
#         }
#         return self.callable_target(**vars_to_substitute)

# class Executor:

#     def __init__(self, tasks: List[Task]):
#         self.tasks = tasks
#         self.leaf_tasks = self.tasks[0].trace_leaf_tasks([])

#     def execute(self)

from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
from typing import Callable, List, Any


class Task:
    def __init__(
        self, name: str, func: Callable[[], Any], dependencies: List["Task"] = None
    ):
        self.name = name
        self.func = func
        self.dependencies = dependencies or []

    def run(self):
        print(f"Running task: {self.name}")
        return self.func()


class Executor:
    def __init__(self, tasks: List[Task]):
        self.tasks = tasks
        self.graph = self.build_graph(tasks)
        self.executed_tasks = set()
        self.executor = ThreadPoolExecutor()

    def build_graph(self, tasks: List[Task]):
        G = nx.DiGraph()
        for task in tasks:
            G.add_node(task.name, task=task)
            for dep in task.dependencies:
                G.add_edge(dep.name, task.name)
        return G

    def run(self):
        # Identify tasks with no dependencies (sources in the DAG)
        futures = {}
        for task in self.tasks:
            if not list(self.graph.predecessors(task.name)):
                futures[self.executor.submit(task.run)] = task

        # Process tasks as their dependencies are resolved
        while futures:
            for future in as_completed(futures):
                finished_task = futures.pop(future)
                self.executed_tasks.add(finished_task.name)
                self.schedule_next_tasks(finished_task, futures)

    def schedule_next_tasks(self, finished_task: Task, futures: dict):
        for successor in self.graph.successors(finished_task.name):
            all_deps_finished = all(
                dep in self.executed_tasks for dep in self.graph.predecessors(successor)
            )
            if all_deps_finished:
                task = self.graph.nodes[successor]["task"]
                if task not in self.executed_tasks:
                    futures[self.executor.submit(task.run)] = task


# Example usage
def task1_func():
    print("Executing Task 1")


def task2_func():
    print("Executing Task 2")


def task3_func():
    print("Executing Task 3")


def task4_func():
    print("Executing Task 4")


t4 = Task(name="task4", func=task4_func)
t3 = Task(name="task3", func=task3_func, dependencies=[t4])
t2 = Task(name="task2", func=task2_func)
t1 = Task(name="task1", func=task1_func, dependencies=[t2])

executor = Executor(tasks=[t1, t2, t3, t4])
executor.run()
