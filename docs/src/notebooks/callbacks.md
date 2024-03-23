
!!! tip annotate "Run This Tutorial in Google Colab"

    This document is automatically generated from a Jupyter notebook. You have the option to open it in Google Colab to interactively run and explore the code in each cell.


    <a target="_blank" href="https://colab.research.google.com/github/osinenkop/regelum-control/blob/master/docs/notebooks/collabs/callbacks.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>
# Callbacks in Regelum Framework

## Introduction

In the Regelum framework, entities can be extended with customized behavior through a feature known as callbacks. Each object that inherits from the `RegelumBase` can utilize callbacks by decorating its methods with `apply_callbacks()`. This augments the standard behavior of methods by executing an `on_function_call` method post-invocation. Callbacks in regelum designed mostly for logging and saving data so that it must not modify the output of the function call.

## How Callbacks Work
There are two main method of Callback one might want do define:  `on_function_call` and `is_target_event`.

They both get the following data automatically:

- `obj`: the object instance from which the method is called
- `method`: the name of the method being called
- `output`: the output yielded by the method

### The `on_function_call` Method

The callback’s logic can is intended to perform actions based on this information, such as logging outputs or saving historical data.

### The `is_target_event` Method

The `is_target_event` method enables a callback to discern between method calls.
Using this method, a callback can decide whether to trigger it's `on_function_call` method based on the passed method's context and results.

```python

class MyCallback(Callback):
    def is_target_event(self, obj, method, output):
        # Define logic to decide whether this is the target event
        pass

    def on_function_call(self, obj, method, output):
        # Executing callback logic
        pass
        
```

Let's see an example of a callback that logs the outputs of all methods in an object:


```python
from regelum.callback import Callback
from regelum.__internal.base import RegelumBase, apply_callbacks
from regelum import set_jupyter_env
import time
```


```python
class SimpleClass(RegelumBase):

    @apply_callbacks()
    def greet_world(self):
        return f"Hello, World!"

    @apply_callbacks()
    def greet(self, name):
        return f"Hello, {name}!"
```


```python
class SimpleLoggerCallback(Callback):

    def is_target_event(self, obj, method, output, triggers):
        return isinstance(obj, SimpleClass)

    def on_function_call(self, obj, method, output):
        self.log(output)
```

### Integrate callbacks at runtime

Normally, when we're executing our scripts from .py files, we activate certain callbacks using regelum.main decorator. But since we are in jupyter notebook, we can use a helper function `set_jupyter_env` for that.


```python
set_jupyter_env(callbacks=[SimpleLoggerCallback])
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">[&lt;</span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">__main__.SimpleLoggerCallback</span><span style="color: #000000; text-decoration-color: #000000"> object at </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0x751ecae06050</span><span style="font-weight: bold">&gt;]</span>
</pre>



## Test it out!


```python
our_class_instance = SimpleClass()
our_class_instance.greet_world()
our_class_instance.greet("Mr. Lyapunov")
time.sleep(1)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[17:15:50] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Hello, World!                                                                   <a href="file:///tmp/ipykernel_3622409/1299929562.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1299929562.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///tmp/ipykernel_3622409/1299929562.py#7" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">7</span></a>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Hello, Mr. Lyapunov!                                                            <a href="file:///tmp/ipykernel_3622409/1299929562.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1299929562.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///tmp/ipykernel_3622409/1299929562.py#7" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">7</span></a>
</pre>



## More on callbacks

### Ready-to-use callbacks

There are plenty of implemented callbacks providing a lot of functionality out-of-the-box. More information can be found [here](https://regelum.aidynamic.io/reference/callback/).

### Cooldown

Excessively verbose logging may slow down the program. To avoid this, we can use `cooldown` class property to limit the period of time between two consecutive messages. Measurement is done in seconds. So adjusting it to 0.5 will make sure that no more than one message per half a second will be printed out. Feel free to play around with this value. 

Here goes an example:


```python
SimpleLoggerCallback.cooldown = 1.0
our_class_instance.greet_world()
our_class_instance.greet_world()
print("Let us try to greet Mr. Lyapunov!")
our_class_instance.greet("Mr. Lyapunov")
print(
    "Oh, no! We just called greet method, but it is under cooldown, so nothing happened!"
)
print("But... hold on for a second...")
time.sleep(1)
our_class_instance.greet("Mr. Lyapunov")
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[17:16:01] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Hello, World!                                                                   <a href="file:///tmp/ipykernel_3622409/1299929562.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1299929562.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///tmp/ipykernel_3622409/1299929562.py#7" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">7</span></a>
</pre>



    Let us try to greet Mr. Lyapunov!
    Oh, no! We just called greet method, but it is under cooldown, so nothing happened!
    But... hold on for a second...



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[17:16:02] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Hello, Mr. Lyapunov!                                                            <a href="file:///tmp/ipykernel_3622409/1299929562.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1299929562.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///tmp/ipykernel_3622409/1299929562.py#7" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">7</span></a>
</pre>


