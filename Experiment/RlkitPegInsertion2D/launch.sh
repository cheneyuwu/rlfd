mpirun -n 5 python -m td3fd.launch --targets train:sac_train_rlbc.py ; \
mpirun -n 5 python -m td3fd.launch --targets train:sac_train_rlbc_qfilter.py ; \
mpirun -n 5 python -m td3fd.launch --targets train:sac_train_rl.py ; \
mpirun -n 5 python -m td3fd.launch --targets train:sac_train_gan.py ; \
mpirun -n 5 python -m td3fd.launch --targets train:sac_train_maf.py ; \
mpirun -n 5 python -m td3fd.launch --targets train:td3_train_rlbc.py ; \
mpirun -n 5 python -m td3fd.launch --targets train:td3_train_rlbc_qfilter.py ; \
mpirun -n 5 python -m td3fd.launch --targets train:td3_train_rl.py ; \
mpirun -n 5 python -m td3fd.launch --targets train:td3_train_gan.py ; \
mpirun -n 5 python -m td3fd.launch --targets train:td3_train_maf.py


mpirun -n 3 python -m td3fd.launch --targets train:sac_train_maf.py ; \
mpirun -n 3 python -m td3fd.launch --targets train:td3_train_maf.py ; \
mpirun -n 3 python -m td3fd.launch --targets train:td3_train_gan.py ; \
mpirun -n 3 python -m td3fd.launch --targets train:sac_train_gan.py
