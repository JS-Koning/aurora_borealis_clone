# Weekly progress journal

## Week 1
1. General
    - Background information gathering
    - Runge Kutta 4th order implementation
    - Numba implementation
        - Significant improvement in computation speed
</br>
</br>
2. Review (w.r.t) original plan
    - x
</br>
</br>
3. Things that need improvement
    - Electron implementation
        - Electron behaviour is not as expected, spiral forming not always present.
    - Jim - Install new GUI, current one is not working correctly for 3D plotting

4. Summary
    - We chose to neglect relativity. The initial speeds are not within 1/10 the velocity of the speed of light, thus relativity can be neglected (might change in the future).
    - Beta+ particles move as expected, starting with a low velocity, spiral forming is present. These are shown in the figure below.
    ![](Images/Figure_1.png)
    
    - For Beta- particles, the behaviour is not as expected, probably due to a bug in the force calculations. Spiral forming is not present and thus requires a fix.
    
    - For next week, we expect to have an array of parallel incoming particles, and follow the trajectory of each individual particle with varying incoming speeds. 
</br>
</br>
5. Questions
    - None
</br>
</br>
6. Next weeks milestones

    - [ ] Implement time dependent velocity plots
    - [ ] Implement time dependent distance plots
    - [ ] Extrapolate measured quantities at a distance of (r_earth+variable) 
    - [ ] Initial ReadMe file
    - [ ] Fix electron movement
    - [ ] Data gathering for incoming particles
</br>
</br>
