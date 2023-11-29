{% raw %} 

$$
\state = \left(\begin{array}{c}
       \vartheta \\
        x \\
        \omega  \\
        v_x 
\end{array}\right), \qquad
\obs = \left(\begin{array}{c}
       (\vartheta \text{ mod } 2 \pi) - \pi  \\
        x \\
        \omega  \\
        v_x 
\end{array}\right)
$$

- $\vartheta$ - pole turning angle [rad]
- $x$ - x-coordinate of the cart [m]
- $\omega$ - pole angular speed with respect to relative coordinate axes with cart in the origin [rad/s]
- $v_x$ - absolute speed of the cart [m/s]

{% endraw %}