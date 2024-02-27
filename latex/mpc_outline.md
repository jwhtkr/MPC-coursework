# MPC Pontification

## Paper Outline/Ideas

### Main contribution(s) thoughts
1) Differentially Flat MPC, i.e., general use MPC with linear system and constraints for any differentially flat system. (Question: start with direct application of differentially flat input, or start with hierarchical MPC->trajectory tracking control method?)
2) Used in ROS framework (rviz? Gazebo? Real robots?)
3) Linear approximations to velocity, and curvature constraints (if we can work around the need for a minimum velocity)?
4) Comparison to MPC with linear dynamics and non-linear constraints and full non-linear MPC?

## Next Steps
### What we have
1) Main optimization framework
2) IRIS to get obstacle-free regions
3) Basic ROS framework (viz, simulation, etc.)

### Gaps
1) IRIS data to optimization (Konnor has done at least some of this)
2) Test environment design (What are we testing, where are obstacles, etc.)
3) MPC framework (i.e., ability to update constraints, initial position, etc. of general optimization)
4) Transfer control to vehicle
5) What do we use for the desired trajectory?
