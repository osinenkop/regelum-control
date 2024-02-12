### How Callbacks Work

Callbacks in Regelum can listen for specific events and respond accordingly. For example, they can log the output of a method, record data when an episode of learning finishes, or even alter the behavior of a class by injecting code. The execution of callbacks is managed by a decorator, and they can be designed to handle various events without modifying the main logic of the application.

### Implementing a Custom Callback

To illustrate how you can implement and use a custom callback in the Regelum framework, let's go through a step-by-step example.

#### Step 1: Derive from the `Callback` Base Class

Your custom callback must inherit from the `Callback` class provided by Regelum. This involves implementing specific methods such as `is_target_event` and `perform`.

from regelum import Callback

class MyCustomCallback(Callback):
    def is_target_event(self, obj, method, output):
        # Define the condition to determine if the current event is the target event
        # For example, log when a specific method ('my_method') is called
        return method == 'my_method'

    def perform(self, obj, method, output):
        # Perform your custom logic here
        # For example, log information to the console
        print(f'MyCustomCallback triggered on method {method}')


#### Step 2: Register Your Callback

Once your custom callback class is defined, you need to register it so that it responds to the specific method calls you're interested in. You can do this by calling the `register` method on your callback class.

# Assuming you have an instance of a RegelumBase-derived class called 'my_agent'
MyCustomCallback.register(attachee=my_agent)


#### Step 3: Use the `@apply_callbacks` Decorator

The `@apply_callbacks` decorator is used to annotate methods within a class derived from 'RegelumBase'. This allows registered callbacks to be triggered before and after the annotated method is called.

from regelum import apply_callbacks, RegelumBase

class MyAgent(RegelumBase):
    @apply_callbacks()
    def my_method(self):
        # Method implementation
        pass

# Create an instance of MyAgent and run 'my_method' to see the callback in action
my_agent = MyAgent()
my_agent.my_method()


When `my_method` is called, `MyCustomCallback` will execute its `perform` method before `my_method` completes, enabling you to add custom behavior or side effects.

By following these steps, you can utilize the callback mechanism in the Regelum framework to augment your RL and optimal control applications with additional behavior, logging, or any other functionality you need without intruding into the main logic of your agents or simulations.