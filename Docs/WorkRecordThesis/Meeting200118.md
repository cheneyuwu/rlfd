# RL + IL Research Project
Meeting Summary\
Jan 18th, 2020


## Tasks & TODOs

- For learning from states
    - environments
        - meta world -> see above
            - need to tune parameters!
        - GYM mujoco environments
            - this works with just TD3 need to make the GAN part work later
        - GYM robotics and your own environments! make them work with the pytorch implementation

- For learning from images
    - input:
        - input is set to 84 x 83 x 3 (according to Melissa)
        - mount the camera on the arm
        - back ground changes (for GAN learning)
    - modifications to GAN
        - replicate actions to make the number of dims of action and observation close
    - modifications to TD3
        - CNN 
        - frame stacking
        - sticky actions
    - algorithms to consider other than TD3
        - model ensemble TRPO/MPPI
    - algorithms to compare
        - PlaNet, D4PG (do not compare to it for now), SAC+AE, SLAC
    - environments
        - meta world -> it has some meta learning baselines, but for single environment, check the sister project "garage"
            - does not have baselines for image based environments, need the trick
        - deepmind suite -> some simple environments and image based baselines (PlaNet)

## Progress and Issues

### Debugging Env
- TD3FDReach2D
    - TD3 + BC
    - TD3 + GAN
    - TD3 + MAF
- RlkitReach2D
    - TD3       => running locally with 2 seeds, done! param ok
    - TD3 + BC  => running locally with 2 seeds, done! param ok
    - TD3 + GAN => running locally with 2 seeds, done! param ok
    - TD3 + MAF => running locally with 2 seeds, done! param ok
    - SAC       => running locally with 2 seeds, done! param ok
    - SAC + BC  => running locally with 2 seeds, done! param ok
    - SAC + GAN => running locally with 2 seeds, done! param ok
    - SAC + MAF => running locally with 2 seeds, done! param ok

### Noisy Demo Env
- TD3FDPegInsertion2D
    - OptDemo
        - TD3 + BC
    - Noise1.0
        - TD3 + BC
- RlkitPegInsertion2D
    - OptDemo
        - **Tuning locally**
            - SAC + BC
            - TD3 + BC
                - prm_loss_weight = 1e-2, 1e-4 => running on megalith with 2 seeds, both works, but not very stable compare to td3fd implementation
        - **Tuning on cedar**
            - SAC + MAF
            - SAC + GAN
    - Noise1.0
        - **Tuning locally**
            - SAC + BC
            - TD3 + BC

### Simple Env

- RlkitPegInsertionRandInit - **Tuning on cedar**
    - SAC + BC 
        - completely not working
        - maybe consider running this after peginhole 2d works
        - 
- RlkitPickAndRandInit - **Tuning on cedar**
    - SAC + BC 
        - prm_loss_weight = 1e-2. 1e-4  => running locally with 1 seed
        - prm_loss_weight = 1e-2        => cluster w/ 5 seeds, not working
    - TD3 + BC 
        - prm_loss_weight = 1e-2. 1e-4  => running locally with 1 seed
                                        => running on cluster with 2 seeds

### Hard Env

- DMC Carpole Swing up - **running on beluga**
    - SAC
    - SAC + BC          
    - SAC + BC + QFilter
    - SAC + GAN
    - SAC + MAF
        - first run failed due to NAN issue => try increase reg_loss weight to 1e3 (this run has been deleted)
        - increased reg_loss_weight to 1e3 and reran the experiments


- DMC Carpole Swing up Sparse - **running on beluga**
    - SAC
    - SAC + BC          
    - SAC + BC + QFilter
    - SAC + GAN
    - SAC + MAF
        - based on previous experiences, use regularizer weight 200 instead
        - 

- DMC HalfCheetah - **running on graham**
    - SAC               
    - SAC + BC          
    - SAC + BC + QFilter
    - SAC + GAN
    - SAC + MAF
        - *_1 => failed due to NAN issue => try increase reg_loss weight to 1e3
        - * increased reg_loss_weight to 1e3 and reran the experiments





## Plots for Interim Report

- Reacher 2D
    - 2D sequence of pictures to illustrate the goal maybe with an arrow
    - GAN & NF potential surface
    - A pic of learning curves this can be done lastly

- PegInsertion2D - **running rvl**
    - Noisy Demonstration
        - SAC                 done
        - SAC + BC            done with 1e-4
        - SAC + BC + QFilter  done with 1e-4
        - SAC + GAN           done
        - SAC + MAF           running
        
        - TD3                 done
        - TD3 + BC            done with 1e-4
        - TD3 + BC + QFilter  done with 1e-4
        - TD3 + GAN           done
        - TD3 + MAF           running
    - Optimal Demonstration
        - SAC                 done
        - SAC + BC            done with 1e-4
        - SAC + BC + QFilter  done with 1e-4
        - SAC + GAN           running
        - SAC + MAF           done (may rerun this without NaN)
        
        - TD3                 done
        - TD3 + BC            done with 1e-4
        - TD3 + BC + QFilter  done with 1e-4
        - TD3 + GAN           running
        - TD3 + MAF           done (may rerun this without NaN)

- DMC Carpole Swing up - **running on**
    - SAC                 done
    - SAC + BC            done
    - SAC + BC + QFilter  done
    - SAC + GAN           done
    - SAC + MAF
    
    - TD3                 running
    - TD3 + BC            running
    - TD3 + BC + QFilter  running
    - TD3 + GAN           running
    - TD3 + MAF 

- DMC Carpole Swing up Sparse - **running on beluga**
    - SAC                 done
    - SAC + BC            done
    - SAC + BC + QFilter  done
    - SAC + GAN           done
    - SAC + MAF           

    - TD3                 done
    - TD3 + BC            done
    - TD3 + BC + QFilter  done
    - TD3 + GAN           done
    - TD3 + MAF           

- DMC HalfCheetah - **running on graham**
    - SAC                 done
    - SAC + BC            done
    - SAC + BC + QFilter  done
    - SAC + GAN           done
    - SAC + MAF           running

    - TD3                 done
    - TD3 + BC            done
    - TD3 + BC + QFilter  done
    - TD3 + GAN           done
    - TD3 + MAF           running

- Meta World Environments (reacher, pickplace, hammer, drawer close) - **running on cedar computers**
    - SAC               
    - SAC + BC          
    - SAC + BC + QFilter
    - SAC + GAN
    - SAC + MAF

    - TD3               
    - TD3 + BC          
    - TD3 + BC + QFilter
    - TD3 + GAN
    - TD3 + MAF

-> note: the experiments for hammer and drawer close and pickplace are running



## Notes
- no sparse for cheetah run
- meta world experiments should start
- debug peginhole2d


## Reminder
