Benchmarks
==========

The speedup graphs below represent the latest results (raw form available `here <https://bitbucket.org/cphvb/benchpress/raw/master/results/akira/benchmark-latest.json>`_) of running benchmarks on akira.

All benchmark results are stored in json-format and are available `here <https://bitbucket.org/cphvb/benchpress/raw/master/results>`_.
Benchmarks are run automatically in a daily fashion on akira and marge, results for p31sd and smithers01 are obtained manually.

To compare benchmark-results from different machines and revisions, take a look at the `compare-tool <http://cphvb.org/benchmarks/compare.html>`_ that is where the magic is happens.

.. image:: https://bitbucket.org/cphvb/benchpress/raw/master/graphs/akira/latest/black_scholes_speedup.png
   :align: center
   :alt: "Black Scholes"

.. image:: https://bitbucket.org/cphvb/benchpress/raw/master/graphs/akira/latest/cache_synth_speedup.png
   :align: center
   :alt: "Cache Synth"

.. image:: https://bitbucket.org/cphvb/benchpress/raw/master/graphs/akira/latest/jacobi_iterative_speedup.png
   :align: center
   :alt: "Jacobi Iterative"

.. image:: https://bitbucket.org/cphvb/benchpress/raw/master/graphs/akira/latest/jacobi_iterative___reduce_speedup.png
   :align: center
   :alt: "Jacobi Iterative - Reduce"

.. image:: https://bitbucket.org/cphvb/benchpress/raw/master/graphs/akira/latest/knn_speedup.png
   :align: center
   :alt: "kNN"

.. image:: https://bitbucket.org/cphvb/benchpress/raw/master/graphs/akira/latest/lattice_boltzmann_2d_speedup.png
   :align: center
   :alt: "Lattice Boltzman 2D"

.. image:: https://bitbucket.org/cphvb/benchpress/raw/master/graphs/akira/latest/lattice_boltzmann_3d_speedup.png
   :align: center
   :alt: "Lattice Boltzman 3D"

.. image:: https://bitbucket.org/cphvb/benchpress/raw/master/graphs/akira/latest/monte_carlo_pi___ril_speedup.png
   :align: center
   :alt: "Monte Carlo PI"

.. image:: https://bitbucket.org/cphvb/benchpress/raw/master/graphs/akira/latest/monte_carlo_pi___2xn_speedup.png
   :align: center
   :alt: "Monte Carlo PI - 2xN"

.. image:: https://bitbucket.org/cphvb/benchpress/raw/master/graphs/akira/latest/monte_carlo_pi___nx2_speedup.png
   :align: center
   :alt: "Monte Carlo PI - Nx2"

.. image:: https://bitbucket.org/cphvb/benchpress/raw/master/graphs/akira/latest/shallow_water_speedup.png
   :align: center
   :alt: "Shallow Water"

.. image:: https://bitbucket.org/cphvb/benchpress/raw/master/graphs/akira/latest/stencil___1d_4way_speedup.png
   :align: center
   :alt: "Stencil 1D 4Way"

.. image:: https://bitbucket.org/cphvb/benchpress/raw/master/graphs/akira/latest/stencil___2d_speedup.png
   :align: center
   :alt: "Stencil 2D"

Deploying the Buildbot
======================

Log into the machine you want to run benchmarks on. Then do the following::

    cd ~
    mkdir buildbot
    cd buildbot
    git archive --remote=ssh://git@bitbucket.org/cphvb/benchpress.git HEAD: --format=tar bootstrap.sh -o bootstrap.sh.tar
    tar xf bootstrap.sh.tar
    rm bootstrap.sh.tar
    chmod +x bootstrap.sh
    # Adjust the script to the local environment
    vim bootstrap.sh
    # Execute it to see that it works.
    ./bootstrap.sh

After you have confirmed that the scripts runs without error, inspect the $MACHINE.log file
Then add it to a cron-job or something like that::

    crontab -e

With a line something like::

    01      3       *       *       *       $HOME/buildbot/bootstrap.sh >> $HOME/buildbot/cron.log 2>&1

Auth to repos
-------------

If you do not already have it set up then you need to set up a ssh-agent with keys to the benchpress repos.
U could a script similar to::

    agent_pid="$(ps -ef | grep "ssh-agent" | grep -v "grep" | awk '{print($2)}')"
    if [[ -z "$agent_pid" ]]
    then
        eval "$(ssh-agent)"
        ssh-add
    else
        #agent_ppid="$(ps -ef | grep "ssh-agent" | grep -v "grep" | awk '{print($3)}')"
        agent_ppid="$(($agent_pid - 1))"
     
        agent_sock="$(find /tmp -path "*ssh*" -type s -iname "agent.$agent_ppid")"
     
        echo "Agent pid $agent_pid"
        export SSH_AGENT_PID="$agent_pid"
        export SSH_AUTH_SOCK="$agent_sock"
    fi


