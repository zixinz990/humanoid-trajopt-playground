# Humanoid Trajectory Optimization Playground

This repository contains several Python scripts for solving trajectory optimization problems involving challenging maneuvers on various humanoid robots using whole-body dynamics. Currently, we have:

1. Backflip, simple humanoid (simple_humanoid_backlip.py)
2. Jump, given target landing pose, Talos (talos_jump.py).

![](simple_humanoid_backflip.gif)
![](talos_jump.gif)

More maneuvers will be added! Contributions are welcome!

## Dependencies

These Python packages must be installed: ``pinocchio 3.4.0``, ``crocoddyl 3.0.1``, ``scipy 1.15.3``.

## Notes

### Simple humanoid robot model (VERY IMPORTANT)

We use ``example_robot_data`` to load the simple humanoid model. This package is installed along with ``pinocchio`` and ``crocoddyl``.

However, the model used in this repository is different from the one provided by ``example_robot_data``. To enable backflips, several modifications have been made, including fixing some joints and adding more default joint configurations. The resulting model and configuration file is in ``robots/simple_humanoid``.

To load our customized model, the current solution is a bit of a hack, but it works:
Locate the root directory of the installed ``example_robot_data`` package on your computer, then navigate to the ``robots/simple_humanoid_description`` folder. Inside, you'll find two subfolders: ``srdf`` and ``urdf``. Replace the original files in each folder with our customized SRDF and URDF files.

For example, my Python 3.11.12 interpreter is located at ``~/.pyenv/versions/3.11.12/envs/_traj_opt/bin/python``. The folder mentioned above is at ``~/.pyenv/versions/3.11.12/envs/_traj_opt/lib/python3.11/site-packages/cmeel.prefix/share/example-robot-data/robots/simple_humanoid_description``. There should be only one ``simple_humanoid_description`` folder if you only installed one ``example_robot_data`` package. You can try

```
sudo find / -type d -name simple_humanoid_description
```

We did this because it is the easiest way to load a customized robot model for ``crocoddyl``. We know this approach is not elegant, and we will try to find a better solution.
