Configs and basic ``rcognita`` applications
===========================================

RL and control theory are infamous for having overwhelmingly many
entities to keep track of: agents, environments, models, training routines,
integrators, predictors, observers, optimizers... Each of the above in turn
has a number of parameters of its own, and to make things worse,
your setup will most likely be highly sensitive to all of these. Therefore
tweaking and tuning your setup may and will get tedeous unless you figure
out a way to do it conveniently and systematically.

Enter hierarchical configs! Rcognita has a builtin hierarchical config pipeline
built on top of ``hydra``. It must be noted that a regular ``hydra``
config will run on ``rcognita`` just fine (but not vice versa), since
``rcognita`` includes all of the original features and syntaxes of ``hydra``.
However ``rcognita`` additionally provides convenient syntactic sugars that
``hydra`` does not posses.

This tutorial is focused on explaining the structure and intended workflow of
``rcognita`` from the perspective of the config paradigm.

The reader is encouraged to familiarize themselves
with ``hydra``. Some of the basic syntaxes along with the additional features will also
be covered in the present tutorial.

The first two examples may have been reused elsewhere in the documentation, so
be sure to skip them if you've already seen them.

Example 1: Basics
-----------------
Consider the following files in your hypothetical project.

``my_utilities.py``:
::

    from rcognita.systems import System
    from rcognita.controllers import Controller

    class MyRobotSystem(System):
        def __init__(self, x, y, z):
            ...

        def ...

    class MyAgent(Controller):
        def __init__(self, a, b, c):
            ...

        def ...


``my_config.yaml``:
::

    rate: 0.1

    initial_state: = numpy.zeros(5) # The '=' lets us evaluate this
                                    # python code 'numpy.zeros(5)'

    robot:
        _target_: my_utilities.MyRobotSystem # '_target_' is a special field
        x: 1                                 # that should point to a class
        y: 2
        z: 3

    agent:
        _target_: my_utilities.MyAgent
        a: 3
        b: 4
        c: 5

``main.py``:
::

    import rcognita as r
    from rcognita.simulator import Simulator
    from rcognita.scenarios import EpisodicScenario
    import my_utilities
    import numpy


    @r.main(
        config_path=".",
        config_name="my_config",
    )
    def my_app(config):
        robot = ~config.robot      # '~' instantiates the object
        controller = ~config.agent # described in the corresponding
                                   # field. It makes use of '_target_'.
        simulator = Simulator(robot,
                              config.initial_state,
                              sampling_time=config.rate)
        scenario = EpisodicScenario(simulator, controller)
        scenario.run()


    if __name__ == "__main__":
        my_app()

The above example project is the equivalent to the first example in section
"What is ``rcognita``?". Here instead of providing args for
MyRobotSystem and MyAgent inside the python script, we instead specify
both the classes and their args in ``my_config.yaml``.

Note, that the operator ``~`` is necessary to let rcognita know that
the corresponding node within the config describes an instance of a class
and we would like to instantiate it
(as opposed to accessing it as a config-dictionary).

In other words ``~config.robot`` evaluates to
::
    <my_utilities.MyRobotSystem object at 0x7fe53aa39220>

, while ``config.robot`` evaluates to
::

    {'_target_':'my_utilities.MyRobotSystem', 'x':1, 'y':2, 'z':3}

Example 2: Nested instantiation
-------------------------------
Note, that when using this config paradigm nothing impedes us from instantiating
**literally everything** directly inside the config, leaving the python script
almost empty. Here's an example of how this can be done:

``my_utilities.py``:
::

    from rcognita.systems import System
    from rcognita.controllers import Controller

    class MyRobotSystem(System):
        def __init__(self, x, y, z):
            ...

        def ...

    class MyAgent(Controller):
        def __init__(self, a, b, c):
            ...

        def ...


``my_config.yaml``:
::

    _target_: rcognita.scenarios.Scenario

    simulator:
        _target_: rcognita.simulator.Simulator
        system:
            _target_: my_utilities.MyRobotSystem
            x: 1
            y: 2
            z: 3
        initial_state: = numpy.zeros(5)
        sampling_time: 0.1

    controller:
        _target_: my_utilities.MyAgent
        a: 3
        b: 4
        c: 5

``main.py``:
::

   import rcognita as r
   import my_utilities
   import numpy


    @r.main(
        config_path=".",
        config_name="my_config",
    )
    def my_app(config):
        scenario = ~config
        scenario.run()


    if __name__ == "__main__":
        my_app()

This way of doing it has numerous advantages. Notably, you can now
conveniently override any input parameters, when running the script like so
::

    python3 main.py controller.a=10

or even

::

    python3 main.py simulator._target_=MyOwnBetterSimulator


Remark on overriding and forwarding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Sure, overriding individual parameters is nice, but writing
``simulator.system.x=3`` can be rather inconvenient as compared to simply writing
``x=3``.

Fortunately, ``rcognita``'s configs have a feature that allows you
to **forward** a variable by simply adding the following line to you config:
::

    @simulator.system.x

This way you can simply write
::

    python3 main.py x=10

and this will have the same effect as:
::

    python3 main.py simulator.system.x=10

Forwarding is intended to provide convenience for overriding the select few important
parameters that may be deeply nested. Don't overuse it though or you'll lose the
advantages of a hierarchical structure.

Example 3: Config groups
------------------------
Sure, we can override a parameters or two, but what if we came up against a case
when we want to be able swap out an entire agent or an entire environment without
rewriting the whole config?

Consider the following example:


``my_utilities.py``:
::

    from rcognita.systems import System
    from rcognita.controllers import Controller

    class MyRobotSystem(System):
        def __init__(self, x, y, z):
            ...

        def ...

    class MyAgentReliable(Controller): ## You already know this one works
        def __init__(self, a, b, c):
            ...

        def ...

    class MyAgentExperimental(Controller): ## Perhaps this one works even better
        def __init__(self, e, f, g, h):
            ...

        def ...

``my_config.yaml``:
::

    _target_: rcognita.scenarios.Scenario

    defaults:
        - controller: reliable

    simulator:
        _target_: rcognita.simulator.Simulator
        system:
            _target_: my_utilities.MyRobotSystem
            x: 1
            y: 2
            z: 3
        initial_state: = numpy.zeros(5)
        sampling_time: 0.1

``controller/reliable.yaml``:
::

    _target_: my_utilities.MyAgentReliable
    a: 4
    b: 5
    c: 6


``controller/experimental.yaml``:
::

    _target_: my_utilities.MyAgentExperimental
    e: 7
    f: 8
    g: 9
    h: 10


``main.py``:
::

   import rcognita as r
   import my_utilities
   import numpy


    @r.main(
        config_path=".",
        config_name="my_config",
    )
    def my_app(config):
        scenario = ~config
        scenario.run()


    if __name__ == "__main__":
        my_app()

In the above project we are looking two alternative agents (controller):
the first one called ``MyAgentReliable`` and the other called ``MyAgentExperimental``.

Observe the ``default`` syntax in ``my_config.yaml``.  The line ``- controller: reliable``
makes it so that the node ``config.controller`` is populated by the contents of
``controller/reliable.yaml``. In this case if you wanted to instead try out the
experimental agent (described by ``controller/experimental.yaml``) you would simply
need to execute the following:
::

    python3 main.py controller=experimental

As simple as that!

Note that the directory ``controller`` matches the name of the node it populates.

Additional remarks on defaults
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Consider the following:

``config.yaml``:
::

    defaults:
        - file

The above code will populate ``config.yaml`` with the contents of ``file.yaml``.
This feature can help you avoid a lot of unnecessary
rewriting and duplication.

Also, if you want to override a nested config group you need
to use ``/`` instead of ``.``. For instance, like so
::

    python3 main.py controller/something_inside_my_controller=experimental

Yes, this syntax is a bit strange since some of the things in between those ``/`` may
not even be actual directories, but rather just names of nodes. This however lets
``hydra`` distinguish between overriding a variable and overriding a config. So if you
were to instead execute
::

    python3 main.py controller.something_inside_the_controller=experimental

this would simply assign the string ``"experimental"`` to
``config.controller.something_inside_my_controller`` as opposed to swapping out the
respective config file to ``experimental.yaml``.

Example 4: Instantiating, referencing and inlining
--------------------------------------------------

Instantiation (``~``)
^^^^^^^^^^^^^^^^^^^^^

Imagine the following: you are building an agent that explicitly
accounts for the error of the simulations (for the purpose of improving offline learning).
To be able to extract the necessary data it needs to have access to the simulator instance.
Here's how you could go about doing it.

``my_utilities.py``:
::

    from rcognita.systems import System
    from rcognita.controllers import Controller

    class MyRobotSystem(System):
        def __init__(self, x, y, z):
            ...

        def ...

    class MyAgent(Controller):
        def __init__(self, simulator, a, b, c):
            self.simulator = simulator
            ...

        def ...


``my_config.yaml``:
::

    _target_: rcognita.scenarios.Scenario

    simulator:
        _target_: rcognita.simulator.Simulator
        system:
            _target_: my_utilities.MyRobotSystem
            x: 1
            y: 2
            z: 3
        initial_state: = numpy.zeros(5)
        sampling_time: 0.1

    controller:
        _target_: my_utilities.MyAgent
        simulator: ~ simulator  ## This is where the magic happens.
        a: 3                    ## '~' instantiates config.simulator.
        b: 4                    ## Furthermore, this is going to be the exact
        c: 5                    ## same instance that is produced by ~config.simulator
                                ## when run in python


``main.py``:
::

   import rcognita as r
   import my_utilities
   import numpy


    @r.main(
        config_path=".",
        config_name="my_config",
    )
    def my_app(config):
        print(~config.simulator is config.controller.simulator)    ## Will output True
        print((~config).simulator is config.controller.simulator)   ## Will output True


    if __name__ == "__main__":
        my_app()


In configs ``~`` does the exact same thing that it does in Python: instantiates
an object described by a config node (with a ``_target_``).

The most important aspect of this feature is your ability to assign
different references of the same instance. By default, ``~`` will create
a reference to the same object that is created during recursive instantiation,
when, for instance, running ``~config``. You can however insist on creating your own
distinct instance by using
::

    field: my_instance_name ~ other.field

``field: ~ something`` is short for ``field: ~{something}``.

Reference (``$``)
^^^^^^^^^^^^^^^^^

You can use ``$`` to reference other fields within the config. For
instance if the ``MyAgent`` only needs to know the absolute tolerance ``atol``
to account for the accuracy of simulation, then one could implement that in
the following way:

``my_utilities.py``:
::

    from rcognita.systems import System
    from rcognita.controllers import Controller

    class MyRobotSystem(System):
        def __init__(self, x, y, z):
            ...

        def ...

    class MyAgent(Controller):
        def __init__(self, atol, a, b, c):
            ...

        def ...


``my_config.yaml``:
::

    _target_: rcognita.scenarios.Scenario

    simulator:
        _target_: rcognita.simulator.Simulator
        system:
            _target_: my_utilities.MyRobotSystem
            x: 1
            y: 2
            z: 3
        initial_state: = numpy.zeros(5)
        sampling_time: 0.1
        atol: 0.001

    controller:
        _target_: my_utilities.MyAgent
        atol: $ simulator.atol  ## This will insert 0.001
        a: 3
        b: 4
        c: 5


``field: $ something`` is short for ``field: ${something}``. You can make use of that,
whe you want to compose something out of different fields. For instance:
::
   a: 1
   b: 2
   a_plus_b: = ${a} + ${b}

One could also use ``$`` to references entire config nodes. For instance:
::

    stuff:
       _target_: builtins.dict
       x: 1
       y: 1
    identical_to_stuff: $ stuff ## will be populated with contents of "stuff"

This will however result in ``stuff`` and ``identical_to_stuff`` being
different instances. I.e. ``~config.stuff is ~config.identical_to_stuff`` will
evaluate to ``False``.

``$`` is used for absolute references, while ``$$`` is used for relative references.

Inline (``=``)
^^^^^^^^^^^^^^

This one is pretty simple. All it does is it executes Python code. Make sure
that the relevant modules are imported in your ``main.py``.

``config.yaml``:
::

    pi: = numpy.pi

``main.py``:

    import numpy as np
    ...

``field: = something`` is short for ``field: ={something}``.

Callbacks and Logging
=====================

In ``rcognita`` we avoid mixing logging routines with functional
code. This is motivated by the fact that different applications
may require very different logging behaviors from same objects.
We thus introduce a lightweight event handling system that allows
for flexible and convenient configuration of logging: Callbacks.

A callback is a callable equipped with a logger and an even handling routine.

Let's write a callback the logs the objective every time it's computed.
``my_callbacks.py``:
::

    class ObjectiveCallback(Callback):
        def perform(obj, method, output):
            if method == 'objective':
                self.log(f"The current objective is equal to {output}.")

To make this callback work we would need
to decorate the ``objective`` method with ``@rcognita.callbacks.apply_callbacks`` after
decorating the respective class with ``@rcognita.callbacks.introduce_callbacks()``. This will
make ``objective`` trigger callback events. To make sure the event is handled we will also need to
register the callback. There are two way of doing so.

You could either specify it in your Python script
``main.py``
::

   import rcognita as r
   import my_callbacks
   import numpy


    @r.main(
        callbacks=[my_callbacks.ObjectiveCallback], ## Do not instantiate it
        config_path=".",
        config_name="my_config",
    )
    def my_app(config):
        ...


    if __name__ == "__main__":
        my_app()

or you could just write the following
to your config:
``config.yaml``
::

    callbacks:
        - my_callbacks.ObjectiveCallback

    ...

The above will not mess with your instantiation parameters. ``config.callbacks`` gets
deleted automatically as soon as ``rcognita`` extracts the callbacks from it.

By default ``rcognita`` creates its own ``Logger`` instance and passes it to the callbacks.
You can insist on your own logger with ``@r.main(logger=..., ...)``.

If you'd like to know more, be sure to read the :doc:`relevant API Docs </modules/rcognita.callbacks>`.

What if I still don't know what I'm doing?
==========================================

If after reading these tutorials you still don't quite know where to start,
do not be discouraged. You are now well equipped to understand the presets provided in
``rcognita``'s repository. As soon as you examine a few of them, you should be able to write
code of your own. In fact many of the presets can likely be conveniently repurposed for
your own projects.

Be sure to hit the API Docs when in doubt, and good luck with your experiments!

