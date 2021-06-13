# Weekly progress journal

## Week 1
1. General
    - Background information gathering
    - Basic simulation implementation
        - Magnetic field
        - Earth (with tilt)
        - Particle system
    - Runge Kutta 4th order implementation
        - For errors smaller than 5th order
    - Numba implementation
        - Significant improvement in computation speed
    - 3D plots
        - Earth model
        - Particle trajectories
</br>
</br>
2. Review (w.r.t) original plan
    - Nothing
</br>
</br>
3. Things that need improvement
    - Electron implementation
        - Electron behaviour is not as expected, spiral forming not always present.
    - Jim - Install proper IDE, current one is not working correctly for 3D plotting

4. Summary
    - We chose to neglect relativity. The initial speeds are not within 1/10 the velocity of the speed of light, thus relativity can be neglected (might change in the future).
    - Beta- particles move as expected, starting with a low- to high solar velocity (250km/s-2000km/s), which is a non-relativistic speed, spiral forming is present. These are shown in the figure below.
    
    ![](Images/proj3_fig1.png)
    
    - The figure below shows a better 3D behaviour, with a different starting location.
    
    ![](Images/proj3_fig2.png)
    
    - For Beta- particles, the behaviour is not as expected, probably due to a bug in the force calculations. Spiral forming is not present and thus requires a fix.
    - Some particles, with a high velocity field will get deflected by the earths magnetic field, just as expected, the velocities plot over time is shown in the figure below, please note that for the second image, the time is not properly shifted.
    
    ![](Images/proj3_fig3.png)
    
    ![](Images/proj3_fig4.png)
    
    - This shielding of the earth is to be expected. A goal is to find specific initial location(s) ranges and velocity ranges where this does not happen. The code is written in such a way, that when the velocity reaches a very low value, the simulation is aborted to increase computing speed. This is because stationary particles are not of interest.
    - A beautifull image is shown below with various trajectories.
    
    ![](Images/proj3_fig5.png)    
    
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

## Week 2
1. Answer on feedback on last week's progress update:
    - I'm confused about some of your remarks: once you say "beta particles behave as expected" and then "beta particles have a bug". What is true?
        - Both are true; we can see that the beta particles (electrons) follow a spiralling trajectory around magnetic field lines (so they behave as expected)
        - However, the 'bug' is that the particles did not seem to reach the Earth (unstable trajectories) due to the low resolution in time-steps (dt).
        - We fixed this 'bug' by increasing the resolution of time-steps (dt).
        - It was not entirely a 'bug' however, since a lower resolution in time-steps would lead to more unstable trajectories; therefore, these particles would (possibly) never be able to reach close to the Earth.
    - In general, I would really like to see a more explicit validation. How do you know your trajectories are ok? Can you compare to a paper?
        - https://folk.ntnu.no/anderhja/aurora/aurora.pdf
            - This paper shows similar spiraling trajectories (although our particle' trajectories are not shown in the Van Allen radiation belt but only in the polar cusp(s)).
        - https://ase.tufts.edu/cosmos/view_chapter.asp?id=29&page=1
            - "An energetic charged particle moves around the magnetic fields in a spiral path that becomes more tightly coiled in the stronger magnetic fields close to a magnetic pole. The intense polar fields act like a magnetic mirror, turning the particle around so it moves back toward the other pole."
            - Our particles can get deflected as well (into the Van Allen radiation belt) for longer simulation times

2. General progress update
    - Information gathering
        - No positrons (Sun does not emit those as much as electrons)
        - Only electrons (no protons, no alpha particles => these are not interesting for Aurora Borealis)
        - Gathered information for coming week
            - Electron absorption in atmosphere (640 km ~ 60 km altitude)
            - 640 km = approx. 0.1 R_Earth, 60 km = approx. 0.01 R_earth
            - Red light emission at highest altitudes, green light around 160 km, blue and purple at lowest altitudes
            - Green light has highest probability
            - Altitude for where electron is absorbed is dependent on energy of electron (higher energy can pass further to reach lower altitudes)
    - Extended simulation implementation
        - Changed Earth's axial tilt (so North Pole collects more particles)
        - Simulation is no longer cut off (unlike before) when a condition is met (not moving towards the Earth any more)
        - Multi-threaded simulation (from 1 to 16 times faster than single-threaded)
        - Numpified and Numbafied simulation
        - Implemented custom grid to improve results
    - Simplified simulation implementation
        - The simulation will focus on the North Pole only due to the low amount of particles we are working with
        - We are neglecting particles returning to the North Pole through the Van Allen radiation belt (simulation time would be too long)
        - We will only work with directly incoming particles that enter through the polar cusp on the North pole side.
        - We create our custom grid so we can collect as much particles as possible in the atmosphere of the Earth (realistically the majority of the particles are deflected by the Earth's magnetic field as expected)
    - Datasets
        - Generate datasets with end trajectories (ending closest to Earth)
        - End trajectories are cut off to save space
        - Datasets are compressed to save space
        - Initial datasets have been created (10000 (100x100) and 32400 (180x180) particles)
    - 3D plots
        - Only relevant trajectories are plotted now (Particles ending closer than 1.1 * R_Earth)
        - Earth axial tilt has been changed into opposite direction so the North pole attracts more particles than the South pole.
    - Code
        - Cleaned up code
        - Optimized code
        - Added comments

3. Review (w.r.t) original plan
    - We could have made more progress on finalizing the project first instead of optimizing the project...
    - [x] Implement time dependent velocity plots
    - [x] Implement time dependent distance plots
    - [ ] Extrapolate measured quantities at a distance of (r_earth+variable)
        - Instead we improved simulation resolution in time which gave expected results
    - [ ] Initial ReadMe file
        - Moved to coming week
    - [x] Fix electron movement
    - [x] Data gathering for incoming particles

4. Things that need improvement
    - Electron absorption should be implemented
    - Reduce dataset file-size (maybe discard all non-interesting particle trajectories)
    - Functions need descriptions (and comments)
    - Minor PEP8 errors are present
    - ReadMe file
    - Optional: We could implement simple relativistic effects
    - Optional: Adaptive Runge-Kutta implementation (too much work for this little time probably)
        - Higher dt steps further away from Earth, lower dt steps closer to Earth

5. Summary
    - Beta- particles move as expected, starting with a low- to high solar velocity (300km/s-750km/s near-Earth):
    
    ![](Images/Figure_18.png)
    
    - This shielding of the Earth is to be expected, which means most simulated particles will be rendered useless:
     
    ![](Images/Figure_21.png)
    
    - Before the particles would go to either the North or South pole, but since we wanted to focus on the North pole only, we adjusted our grid and parameters such to get the following:
    
    ![](Images/Figure_24.png)    
    
    - We also removed positrons in our simulations so we get one polar cusp per pole (as expected), see top and bottom image for comparison.
    
    - Particles currently are able to go through the surface of the Earth (as the particles are not yet absorbed in the atmosphere); this is expected behaviour without absorption.
    
    - Simulations have been heavily optimized by using Numpy, Numba and multi-threading/parallelization; and it might be possible to extend Numba-fication.
    
    - We generated initial datasets to work with.
    
    - We gathered information to finalize our project coming week.

6. Questions
    - None

7. Next weeks milestones
    - [ ] Create animated trajectories
    - [ ] Implement electron absorption
    - [ ] Create plots for latitude vs. absorption
    - [ ] Initial ReadMe file
    - [ ] Finalization of project
    - [ ] Start preparing presentation
    - [ ] Other points mentioned in point 4 (points of improvement)

## Week 3
1. Answer on feedback on last week's progress update:
   - There is one main message I would like to give to you: right now, you are mainly talking about particle trajectories and animations. Those are fine and nice to see, but it's not a very systematic analysis. Make sure you do a proper statistical analysis. Not sure what is usually done, but you could do things such as probability of reflection depending on the incoming offset, or incoming particles per latitude, etc.
      - We thought about probability of reflection, but due to the size of our system and the 'size' of an electron this was not possible.
      - However, we did do some analysis for the incoming particles combined with their height of absorption to create aurora's. However, during our research we found out that this is extremely complex in the given timeframe, thus we introduced a simplified method.
      - We could still create a lattitude based figure, this would not take much time to implement due to the method we used to create datasets.
         - Currently, we do have a 3D plot of the locations where the particle would collide with other particles in the atmosphere. This might not be systematic enough for a report, but it is in our opinion the best visualization for a presentation. However, as stated before, a latitude based figure could be created+presented as well.

2. General progress feedback:
   
   - Finalized code
      - Finalized simulation datasets
         - Added datasets of particles hitting earth trajectories
      - Added method for absorption of the electrons in the atmosphere.
           - This however, was created without scattering and secondary absorption. Instead, it was gathered using data obtained from the nasa.gov site, focussing on a single absorption of an electron. 
            - We have found a [great paper](http://pedrinho.cs.uwaterloo.ca/~gvgbaran/cgpub/aurora1.pdf) for absorption, however it does not allow implementation in our current simulation setup.
      - Finalized most images
         - Some more might be created for the presentation
      
    - Limitations of our simulation.
        - Our simulations has a serious limitation. In the Van Allen belt, our particles do not speed up, and start to slow down even. The reason why: It is due to the limitation in the physics part of the the code. What is wrong: We did not take in to account 'plasma wave interactions'. Some reading up we did are the following articles: 
            - [general Wikipedia](https://en.wikipedia.org/wiki/Two-stream_instability)
            - [A tutorial presentation of the two stream instability and Landau damping](http://people.na.infn.it/~fedele/TESINE/Landau%20damping/Anderson%20et%20al_Am%20J%20Phys_69_1262_2001.pdf)
            - [Statistical Distribution of Whistler Mode Waves in the Radiation Belts With Large Magnetic Field Amplitudes and Comparison to Large Electric Field Amplitudes](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019JA026913)
            - [Formation process of relativistic electron flux through interaction with chorus emissions in the Earth's inner magnetosphere](https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2015JA021563)
        - Our current understanding of the limitations is as following:
            - The plasma, which the Van Allen belt consists of, a soup of charged particles creates opposite moving streams (ions and electrons), will behave like waves, creating electromagnetic pulses troughout the plasma waves. These will either give the (relative lower mass) electrons a sudden(?) high kinetic energy of __at least__ 35 times higher than our current simulation. 
            - Not only plasma waves do this, but also electrons without ions (with different speeds and/or densities etc), its a very complex interaction with multiple components.
        - How are we planning to fix this?
            - Well, we're not really going to fix this physically. The mathematics are too complex to implement in the given timeframe.
            - ~~What we're instead going to do is simply boosting the velocities of particles 'stuck' in the Van Allen belt to a high velocity, and seeing what is going to happen to them. Is this physically correct? Not really, but it is the result of a physical phenomena. We did read that, the speedboost of the electrons are in line with the existing magnetic field lines. So, in our simulation we're skipping a simulation-step and using direct results of other papers.~~
            - Were going to speed up the particles that we know are going to hit the earth. Skipping the exact physical phenomena, and going directly to (approximated) result of the other papers.
   




3. Review (w.r.t.) original plan
    - [x] Create animated trajectories
    - [x] Implement electron absorption
    - [ ] Create plots for latitude vs. absorption
      - [x] Added 3D view instead
    - [ ] Initial ReadMe file
    - [x] Finalization of project
    - [x] Start preparing presentation
    - [x] Other points mentioned in point 4 (points of improvement)
   
4. Things that need improvement
   - None that will be implemented before the final submission
   
5. Summary
   - Multiple new animations are created to focus on presentations, a speedup is present to match emperical/theoretical data. An example is shown below. An animation is higly useful to allow visualization of velocity.
   
     ![](Images/Animation_t0.001dt1e-08n32400.gif)

   - Absorption is implemented, the results are shown in the figure below. Note that a simplified method is applied. The scale and complexity is simply too high to implement the method we wanted to.
   
     ![](Images/Figure_38.png)

   - A visual view of this absorption view, including colors and heights are shown in the figure below.
   
     ![](Images/Figure_36.png)

   - This can be summarized in the following bar chart. Here, the y-axis is the height of absorption
   
     ![](Images/Figure_37.png)

   - Probabilities are present in absorption, these are shown in the plot below, including uncertainties. However, they are hard to inteprent, not all particles are able to absorb due to not having their end trajectories penetrating earth, we have som stray particles, and the absorption is not a continous process, but a discrete one (due to the emperical data used as atmospheric data). The particles with zero uncertainty are the particles that get deflected back into the van Allen belt, these particles however do not have a zero uncertainty of absorption in the real life scenario.
     ![](Images/Figure_errorbars.png)
     
   - For correctness checks, we will refer to the presentation that will be uploaded tomorrow. These will have some visualization comparing the obtained data from the simulation to other simulations, or real life empirical data.
   
6. Questions
    - None