"""Configuration used to demonstrate running Parsl on ALCF's Theta supercomputer
connecting from your local system.

WARNING: You should probably use FuncX if you are looking for remote access,
or launch your application from the Theta login node if you want to use Parsl.

Before running, you must open ports on the remote system or provide an SSH tunnel.
ALCF prefers SSH tunnels, which we open using

```bash
ssh -R 54928:localhost:54928 -R 54875:localhost:54875 lward@thetalogin6.alcf.anl.gov
````

This makes it such that processes on thetalogin6 can connect to ports on my local system by
connecting to the matching ports on that system.
Processes on remote systems (i.e., compute nodes) cannot access these ports on the login nodes
due to ALCF's security policies.
Allowing access to the ports on the login node (thetalogin6) can be accomplished by adding another
SSH tunnel from one of the MOM node, which lie on the same network as compute nodes.

First log in to thetamom1 from the login node and set up local port forwarding bound to `0.0.0.0`
so compute nodes that use it to forward requests

```bash
ssh thetamom1
ssh -v -N -L 0.0.0.0:54928:localhost:54928 -L 0.0.0.0:54875:localhost:54875 thetalogin6
```

You now will have secure channels between a Theta compute node and your home computer,
 routed through the Theta login and MOM nodes.

Note that we specify the same service nodes (thetalogin6, thetamom1) throughput the input file.
"""

from parsl.executors import HighThroughputExecutor
from parsl.providers import CobaltProvider
from parsl.launchers import AprunLauncher
from parsl.channels import SSHInteractiveLoginChannel
from parsl import Config


def theta_remote() -> Config:
    """Configuration where the manager sits on a local machine and you SSH into Theta

    Returns:
        Parsl configuration
    """
    # Set a Theta config for using the KNL nodes with 8 workers per node
    scr_dir = '/lus/theta-fs0/projects/CSC249ADCD08/molecular-design-parsl-demo/parsl-dir'
    config = Config(
        executors=[
            HighThroughputExecutor(
                address='thetamom1',  # Workers will connect back to thetalogin6
                label='knl',
                max_workers=8,
                worker_ports=(54928, 54875),  # Hard coded to match up with SSH tunnels
                # Mark where we can write log files
                worker_logdir_root=scr_dir,
                provider=CobaltProvider(
                    channel=SSHInteractiveLoginChannel(
                        'thetalogin6.alcf.anl.gov', username='lward',
                        script_dir=scr_dir
                    ),
                    queue='debug-flat-quad',  # Flat has lower utilization
                    account='redox_adsp',
                    launcher=AprunLauncher(overrides="-d 64 --cc depth -j 1"),
                    worker_init='''
module load miniconda-3
source activate /lus/theta-fs0/projects/CSC249ADCD08/molecular-design-parsl-demo/env
which python
''',
                    nodes_per_block=8,
                    init_blocks=0,
                    min_blocks=0,
                    max_blocks=1,
                    cmd_timeout=300,
                    walltime='00:60:00',
                    scheduler_options='#COBALT --attrs enable_ssh=1'
                ))
        ]
    )
    return config
