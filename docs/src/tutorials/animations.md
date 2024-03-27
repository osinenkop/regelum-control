### Animations are callbacks

Animations in regelum are [callbacks](../../notebooks/callbacks) that collect and plot data.
To make some animation activate when a particular [system](system.md) is used, you need 
to simply "attach" it:
```python
@MyAnimation.attach
class MySystem(System):
    ...
```

or equivalently:

```python
MySystem = MyAnimation.attach(MySystem)
```

There are two ways to attach multiple animations:

```python
MyComposedAnimations = MyFirstAnimation + MySecondAnimation

@MyComposedAnimations.attach # The first way is to add
class MySystem(System):      # your animations and attach their sum.
    ...

@MyFirstAnimation.attach     # The second way is to simply make
@MySecondAnimation.attach    # multiple attachments.
class MyOtherSystem(System):
    ...

```


Most built-in systems that come with regelum already have some animations attached to them.
If you want to detach them use [``callback.detach``][regelum.callback.detach]. For example:

```python
from regelum import callback

class MyNonAnimatedSystem(System): # This system is not animated
    ...

# The system below is animated with MyAnimation.
MyAnimatedSystem = MyAnimation.attach(MyNonAnimatedSystem) 


# The system below is no longer animated. 
# It is exactly the same as MyNonAnimatedSystem.
MyOtherNonAnimatedSystem = callback.detach(MyAnimatedSystem)

```

Detaching is especially useful when deriving from built-in systems:

```python
from regelum import callback
from regelum import system

@MyCustomAnimation.attach
@callback.detach
class MyCustomRobot(system.ThreeWheeledRobotKinematic):
    ...

```

### Making a single-body animation

Regelum lets you easilly animate a shape moving on a plane. Let's say we
want to animate the motion of three-wheeled robot moving on a plane.
Here's how this can be implemented with regelum, assuming that you are aware of your state variables:

```python
from regelum.system import System
from regelum.animation import SingleBodyAnimation

# Before you start, look at _state_naming of your system.
# This will tell you what the state variables are
# and how they are ordered.
class ThreeWheeledRobot(System):
    _state_naming = ["horizontal coordinate [m]", 
                     "vertical coordinate [m]", 
                     "angle [rad]"]
    ...

# First, derive from SingleBodyAnimation
class ThreeWheeledRobotAnimation(SingleBodyAnimation): 
    _pic = "/path/to/ThreeWheeledRobot.svg" # provide an image of the robot
    
    # Tell where to set the marker depending on the system's state
    def on_trigger(self, _):
        # Let's extract the state variables
        horizontal = self.system_state[0] # location of the robot along x-axis
        vertical = self.system_state[1]   # location of the robot along y-axis
        angle = self.system_state[2]      # angle of the robot along it's own axis
        
        # Now let's set the marker to the correct position
        self.add_frame(x=horizontal,  # location of the marker along x-axis
                       y=vertical,    # location of the marker along y-axis
                       theta=angle)   # rotation of the marker
    

ThreeWheeledRobot = ThreeWheeledRobotAnimation.attach(ThreeWheeledRobot)
```

Take note of the following things:

 * `on_trigger` is being called whenever the system's state is updated
 * `self.system_state` will thus contain the most recent values of state variables
 * `self.add_frame(x, y, theta)` will produce a frame for the animation. Arguments `x`, `y`, and `theta` specify the position of the marker.
 * The visual appearance of the marker is determined by `_pic`.

It is important to point out that the code  for `on_trigger` used in this snippet is exactly the same
as in `SingleBodyAnimation.on_trigger`, thus overriding it was not actually necessary. This means
that this snippet could be simplified to only two lines:
```python
class ThreeWheeledRobotAnimation(SingleBodyAnimation): 
    _pic = "/path/to/ThreeWheeledRobot.svg"
```

There are other class attributes beside ``_pic`` that you can use to tune the appearance
of the marker:
```python
class ThreeWheeledRobotAnimation(SingleBodyAnimation): 
    _pic = "/path/to/ThreeWheeledRobot.svg"
    _rotation = 180     # This will rotate the marker 180 degrees.
    _marker_size = 600  # This will make the marker very big.
```
If you do not yet have an image for your animation, you can simply omit ``_pic``.
In this case the default image will be used for your system, an acute triangle.
```python
# This one will work exactly like ThreeWheeledRobotAnimation,
# but you'll see a trianlge rolling on a plane, rather than a robot.
class MyTriangleAnimation(SingleBodyAnimation):
    pass

# The following definition is equivalent
class MyOtherTriangleAnimation(SingleBodyAnimation):
    _pic = 'default.svg' # This refers to a built-in marker of regelum

```


### Making a multiple-body animation

Making an animation for multiple shapes is a very similar procedure:

```python
from regelum.system import System
from regelum.animation import PlanarBodiesAnimation

class TwoRobots(System):
    _state_naming = ["horizontal coordinate of robot 1 [m]", 
                     "vertical coordinate of robot 1 [m]", 
                     "angle of robot 1 [rad]",
                     "horizontal coordinate of robot 2 [m]",
                     "vertical coordinate of robot 2 [m]",
                     "angle of robot 2 [rad]"]
    ...

# Derive from PlanarBodiesAnimation
class TwoRobotsAnimation(PlanarBodiesAnimation): 
    _pics = ["/path/to/FirstRobot.svg",
             "/path/to/SecondRobot.svg"]
    _rotations = [0, 0]
    _marker_sizes = [50, 50]
    
    # Tell where to set the markers depending on the system's state
    def on_trigger(self, _):
        # Let's extract the state variables
        horizontal_1 = self.system_state[0] # location of the first robot along x-axis
        vertical_1 = self.system_state[1]   # location of the first robot along y-axis
        angle_1 = self.system_state[2]      # angle of the first robot along it's own axis
        
        horizontal_2 = self.system_state[3] # location of the second robot along x-axis
        vertical_2 = self.system_state[4]   # location of the second robot along y-axis
        angle_2 = self.system_state[5]      # angle of the second robot along it's own axis
        
        # Now let's set the marker to the correct position
        self.add_frame(x0=horizontal_1,   # location of the first marker along x-axis
                       y0=vertical_1,     # location of the first marker along y-axis
                       theta0=angle_1,    # rotation of the first marker
                       x1=horizontal_2, # location of the second marker along x-axis
                       y1=vertical_2,   # location of the second marker along y-axis
                       theta1=angle_2   # rotation of the second marker
    

TwoRobots = TwoRobotsAnimation.attach(TwoRobots)
```
To summarize, ``x0``, ``y0``, ``theta0`` represent the coordinates of the first marker, while
``x1``, ``y1``, ``theta1`` represent the coordinates of the second marker. 
Naturally, you can have more than two shapes. For instance if ``len(_pics) == 3``, then
you could add a frame that has three shapes on it:
```python
class ThreeTrianglesAnimation(PlanarBodiesAnimation):
    _pics = ['default.svg'] * 3
    
    def on_trigger(self, _):
       self.add_frame(x0=1, y0=2, theta0=3, 
                      x1=4, y1=5, theta1=6, 
                      x2=7, y2=8, theta2=9)
```

### Naming axes, setting titles, plotting static shapes

Let's say you want to name you axes before starting the animation. Here's how you could go
about it:
```python
class MovingTriangleAnimation(SingleBodyAnimation):
    def setup(self):
        super().setup()
        self.ax.set_xlabel("Horizontal distance")
        self.ax.set_ylabel("Vertical distance")
```

As you might have guessed, ``setup`` is being called each time the animation is started or
restarted. 
This is why using ``__init__`` for this wouldn't work. 
Indeed, ``__init__`` will only run once, but ``setup`` is run before each start/restart.
Thus, you need to use ``setup`` if you'd like the effects to persist between restarts.

Naturally, you can use ``setup`` to do all sorts of other things:
```python
target_coordinates = [0, 0]

class MovingTriangleAnimation(SingleBodyAnimation):
    def setup(self):
        super().setup()
        self.ax.set_xlabel("Horizontal distance") # label y-axis
        self.ax.set_ylabel("Vertical distance")   # label x-axis
        
        self.ax.set_title("My animation of a traingle")
        
        
        # Now let's create a big red spot to denote the target position
        target_x, target_y = target_coordinates
        self.ax.plot([target_x], [target_y], # coordinates of the spot
                     'red',                  # color 
                      ms=35,                 # size
                      marker='o',            # shape (circular spot)
                      zorder=-1))            # send to background
        # Let's also make it so that "Target" is written on top of this spot
        self.ax.text(target_x, target_y,            # coordinates of the text box
                     "Target",                      # the text
                     horizontalalignment='center',  # centering horizontaly
                     verticalalignment='center',    # centering verticaly
                     zorder=-1,                     # send to background
                     ).set_clip_on(True)            # hide this text when it is outside of limits
```

### Setting axis limits
The reason why axes limits are not covered by ``setup`` is because
when you're animating something, you often **want the axes to be dynamic**, i.e.
changing with time as the shapes are moving. 
Here's how this can be accomplished with regelum:
```python
class MovingTriangleAnimation(SingleBodyAnimation):
    def on_trigger(self, _):
        self.add_frame(x=1, y=2, theta=3)
        # The above will append {"x":1,"y":2,"theta":3} to self.frame_data
        # (it does a few other things too)
    
    def lim(self, 
            frame_idx): # ID of current frame
        current_frame = self.frame_data[frame_idx] 
        
        x = current_frame["x"] # let's extract the coordinates
        y = current_frame["y"] # from the frame data that we
                               # added earlier with self.add_frame
        
        # self.lim_from_reference(xs, ys) will construct limits
        # that cover all x-coordinates from xs and all 
        # y-coordinates from ys
        left, right, bottom, top = \
            self.lim_from_reference([x - 2, x + 2], # our limits will span a
                                    [y - 2, y + 2]) # 4x4 square around the triangle
        
        self.ax.set_xlim(left, right) # set limits for x-axis
        self.ax.set_ylim(bottom, top) # set limits for y-axis

```

Here we explicitly interact with ``self.frame_data``, which is a list
of all frames (dicts of parameters for plotting) that we added with 
``self.add_frame(...)``.


### Graph animations

If you want to have an animated graph that grows as new data comes in, first
check out the graph animations that are already available in regelum:

 * [``animation.StateAnimation``][regelum.animation.StateAnimation] will plot the state variables over time
 * [``animation.ObservationAnimation``][regelum.animation.ObservationAnimation] will plot observables over time
 * [``animation.ActionAnimation``][regelum.animation.ActionAnimation] will plot control inputs over time
 * [``animation.ValueAnimation``][regelum.animation.ValueAnimation] will plot episode-average reward/cost attained over iterations

Now, if you found yourself in a situation when you need to plot 
something that isn't on this list, here's an example of how you could do it.
Let's say you want to plot the azimuth of your three wheeled robot:
```python
from regelum import callback
from regelum.animation import GraphAnimation
import numpy as np

class AzimuthAnimation(GraphAnimation,         # functionality for animating graphs
                       callback.StateTracker,  # collects current state and stores in self.system_state
                       callback.TimeTracker):  # collects current time and stores in self.time
    def setup(self):
        super().setup()
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Azimuth")
        self.t = []
        self.y = []

    def is_target_event(self, obj, method, output, triggers):
        return callback.StateTracker in triggers  # this line says that on_trigger will be called 
                                                  # when the state is collected by StateTracker

    def on_trigger(self, _):
        self.t.append(self.time)    # update time sequence
        
        x = self.system_state[0]    # extract state variables
        y = self.system_state[1]    
        azimuth = np.arctan2(x, y)  # compute azimuth
        
        self.y.append(azimuth)      # update azimuth sequence
        
        self.add_frame(
            line_datas=[(self.t, self.y)] # add a frame that plots azimuth vs. time
        )

```

Making graph animations involves explicitly interacting with trackers. This example
makes use of [``StateTracker``][regelum.callback.StateTracker] that collects the current state in stores it in ``self.system_state``,
but regelum also has several other trackers:

 * [``TimeTracker``][regelum.callback.TimeTracker] stores most recent simulation time in ``self.time`` (also used in the example)
 * [``ActionTracker``][regelum.callback.ActionTracker] stores most recent control input in ``self.action``
 * [``ObservationTracker``][regelum.callback.ObservationTracker] stores most recent observation in ``self.observation``
 * [``ValueTracker``][regelum.callback.ValueTracker] stores most recent total reward/cost over an episode in ``self.value``


### Command line arguments
Here's a list of useful command line arguments for working with animations:

 * `--interactive` will open a window that displays the animations in real time.
 * `--save-animations` will save the animations to the hard drive on every epoch
 * `--playback` will save the animations on the very last epoch and then automatically open them in you default browser
 * `--fps=10` will set the target frame rate to 10 frames per second


### Other technical capabilities

#### Trajectory plotting

You may notice that when using animations derived from [`PlanarBodiesAnimation`][regelum.animation.PlanarBodiesAnimation]
and [`SingleBodyAnimation`][regelum.animation.SingleBodyAnimation] a blue trail is displayed behind the moving object.
You can control this behavior by overriding ``increment_trajectory``:

```python
# default behaviour
class MyAnimation(PlanarBodiesAnimation):
    ...
    def increment_trajectory(self, **frame_data):    
        return frame_data["x0"], frame_data["y0"] 

# no trail
class MyAnimation(PlanarBodiesAnimation):
    ...
    def increment_trajectory(self, **frame_data):    
        return 0, 0
        
# trail that follows the second body
class MyAnimation(PlanarBodiesAnimation):
    ...
    def increment_trajectory(self, **frame_data):    
        return frame_data["x1"], frame_data["y1"] 
```

#### Additional attributes of graph animations
Graph animations have several attributes that offer finer control over the appearance
of the graph. 
For instance,  `_legend` lets you label your curves. 
Attribute `_vertices` controls the limit of points that can 
be plotted per frame. 
If the number of points exceeds this limit, the curve will be downsampled.
The color and appearance of the curves can be set with `_line`.
For instance the code below will create an animation that plots two
dashed curves downsampled to 1000 points and labeled as "reference" and "observed":
```python
class MyGraphAnimation(GraphAnimation, ...):
    _legend = ("reference", "observed")
    _vertices = 1000
    _line = '--'
   ...
```

#### Default animations

Systems derived from [`system.System`][regelum.system.System] will have several animations attached to them by default:

 * [`StateAnimation`][regelum.animation.StateAnimation]
 * [`ValueAnimation`][regelum.animation.ValueAnimation]
 * [`ActionAnimation`][regelum.animation.ActionAnimation]

If you don't want them, use [`callback.detach`][regelum.callback.detach].
