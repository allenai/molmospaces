# Examples

## Inverse Kinematics

Just run the following, replace `ROBOT_ID` with one of the identifiers from above:

```bash
python -m molmo_spaces.examples.ex_ik_robots --robot-id ROBOT_ID
```


| Preview | Id |
|---------|----|
| <img src="../../docs/images/robots/ik_franka_droid.gif"/> | `franka-droid` |
| <img src="../../docs/images/robots/ik_i2rt_yam.gif"/> | `i2rt-yam` |


## G1 navigation

This example uses the [SONIC](https://nvlabs.github.io/GEAR-SONIC/) policy for the `Unitree G1` to navigate in the plane. The user can
translate|rotate the waypoint and the whole body controller will handle the request.

```bash
python -m molmo_spaces.examples.ex_g1_navigation
```

![g1-navigation](../../docs/images/robots/navigation_g1.gif)
