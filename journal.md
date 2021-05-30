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
    
    ![](Images/proj3_fig1.png)
    
    - Note the particles have a low starting velocity, not representing a physical solar ejection, however it does show the behaviour correctly.
    - The figure below shows a better 3D behaviour, with a different starting location.
    
    ![](Images/proj3_fig2.png)
    
    - For Beta- particles, the behaviour is not as expected, probably due to a bug in the force calculations. Spiral forming is not present and thus requires a fix.
    - Some particles, with a high velocity field will get deflected by the earths magnetic field, just as expected, the velocities plot over time is shown in the figure below, please note that for the second image, the time is not properly shifted.
    
    ![](Images/proj3_fig3.png)
    
    ![](Images/proj3_fig4.png)
    
    - This shielding of the earth is to be expected. A goal is to find specific initial location(s) ranges and velocity ranges where this does not happen.
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
