#### Pycharm

You have to use  `netstat -lntu` to find open port with IP address to connect to and set it at the top of the file
you're running under `pydevd-pycharm` at the top. Got the following error `Connection to Python debugger failed: Interrupted function call: accept failed`.
This can occur if you have any files that have conflicting naming with a dependency in `pydevd-pycharm` according to StackOverflow. RobustNav repo might have
conflicting .py file names.
