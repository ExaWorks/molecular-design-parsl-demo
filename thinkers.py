"""A collection of thinkers used in our Colmena examples"""

from collections import defaultdict
from functools import lru_cache
from time import perf_counter, sleep
from threading import Lock, Event, Thread
from random import shuffle
from string import Template
from typing import List

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from IPython import display
from ipywidgets import widgets
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolDraw2DSVG
from colmena.models import Result
from colmena.thinker import BaseThinker, event_responder, task_submitter, result_processor
from colmena.thinker.resources import ResourceCounter
from matplotlib import colors


@lru_cache(128)
def _print_molecule(smiles) -> str:
    """Print a molecule as an SVG
    
    Args:
        smiles (str): SMILES string of molecule to present
        atom_id (int): ID number atom to highlight
    Returns:
        (str): SVG rendering of molecule
    """
    # Compute 2D coordinates
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)

    # Print out an SVG
    rsvg = MolDraw2DSVG(100, 100)
    rsvg.DrawMolecule(mol)
    rsvg.FinishDrawing()
    return rsvg.GetDrawingText().strip()


class BatchedThinker(BaseThinker):
    """A thinker which evaluates molecules in batches.
    
    Also includes a dashboard element designed to be displayed in Jupyter
    """

    def __init__(self, queues, n_to_evaluate: int, n_parallel: int,
                 batch_size: int, initial_count: int,
                 molecule_list: List[str],
                 dashboard: widgets.Output):
        """Initialize the thinker
        
        Args:
            queues: Client side of queues
            n_to_evaluate: Number of molecules to evaluate
            n_parallel: Number of computations to run in parallel
            initial_count: Minimum number of simulations to complete before training
            batch_size: Number of computations to complete before starting a new training job
            molecule_list: List of SMILES strings
            dashboard: Link to Jupyter output
        """
        super().__init__(
            queues,
            ResourceCounter(n_parallel, ['simulate', 'train', 'infer'])
        )

        # Store the user settings
        self.n_to_evaluate = n_to_evaluate
        self.initial_count = initial_count
        self.batch_size = batch_size
        self.n_parallel = n_parallel

        # Dashboard elements
        self.dashboard = dashboard
        self.allocation = {
            'train': 0,
            'simulate': 0,
            'infer': 0
        }
        with open('monitor.html') as fp:
            self.template = Template(fp.read())
        self.dash_lock = Lock()
        self.last_updated = perf_counter()

        def _update_loop():
            while not self.done.is_set():
                self._update_dashboard('simulate', 0)
                sleep(1)

        Thread(target=_update_loop, daemon=True).start()

        # Settings that are not user-configurable yet
        self.inference_tasks = max(
            len(molecule_list) // 20000,
            self.batch_size * 2
        )  # Ensure task sizes are large enough to be interesting

        # Create a database of evaluated molecules
        self.database = dict()

        # Keep track of which molecules we have run
        self.already_ran = set()

        # Create a record of completed calculations
        self.simulation_results = []
        self.learning_results = []

        # Create a priority list of molecules, starting with them ordered randomly
        self.priority_list = list(molecule_list)
        shuffle(self.priority_list)
        self.priority_list_lock = Lock()  # Used to prevent 

        # Create a tracker for how many sent and how many complete
        self.rec_progbar = tqdm(total=n_to_evaluate, desc='started')
        self.sent_progbar = tqdm(total=n_to_evaluate, desc='successful')

        # Create some events that mark the status of the workflow
        self.start_update = Event()  # This Event is triggered when enough simulation tasks have completed
        self.task_list_ready = Event()  # Used to mark that the task list is ready to use
        self.task_list_ready.set()  # It is ready when we start the application

        # Assign all resources to simulation to start with
        self.rec.reallocate(None, 'simulate', n_parallel)

    def _update_dashboard(self, task_type: str, change: int):
        """Make changes to the allocation of tasks on the dashboard

        Args:
            task_type: Type of the task that just finished
            change: How much to change the allocation by
        """

        with self.dash_lock:
            # Update the allocation record
            self.allocation[task_type] += change

            # Exit if the HTML was updated recently
            if perf_counter() - self.last_updated < 1:
                return

            # Get the color for each allocation
            #  Store them in a dictionary that will be used for
            sub_dict = defaultdict(str)
            cell_colors = {
                'sim_color': colors.to_rgb('teal') + (self.allocation['simulate'] / self.n_parallel * 0.8,),
                'tri_color': colors.to_rgb('orangered') + (self.allocation['train'] / 1 * 0.8,),
                'inf_color': colors.to_rgb('darkorchid') + (self.allocation['infer'] / self.inference_tasks * 0.8,),
            }
            sub_dict.update(dict((k, colors.to_hex(v, keep_alpha=True)) for k, v in cell_colors.items()))

            # Store the size of the allocation
            sub_dict['sim_count'] = str(self.allocation['simulate'])
            sub_dict['tri_count'] = str(self.allocation['train'])
            sub_dict['inf_count'] = str(self.allocation['infer'])

            # Get the top molecule
            if len(self.simulation_results) > 0:
                best_mol = max(self.simulation_results, key=lambda x: x.value if x.success else -np.inf)
                sub_dict['best_mol'] = _print_molecule(best_mol.args[0])

            # Add in the recent molecules
            for i, r in enumerate(reversed(self.simulation_results[-10:])):
                sub_dict[f'recent_mol_{i}'] = _print_molecule(r.args[0])

            # Compute the success rates
            if len(self.simulation_results) > 0:
                success_thr = 0.55
                most_recent = min(
                    max(25, self.batch_size * 2),
                    len(self.simulation_results)
                )
                sub_dict['total_eval'] = str(len(self.simulation_results))
                success_count = sum(x.success and x.value > success_thr for x in self.simulation_results)
                total_count = len(self.simulation_results)
                sub_dict['total_success'] = f'{success_count} ({success_count / total_count * 100:.0f} %)'
                success_count = sum(x.success and x.value > success_thr for x in self.simulation_results[-most_recent:])
                sub_dict['recent_success'] = f'{success_count} ({success_count / most_recent * 100:.0f} %)'

            # Render the template
            html = self.template.substitute(sub_dict)
            self.dashboard.outputs = []
            self.dashboard.append_display_data(display.HTML(html))
            self.last_updated = perf_counter()

            # Save it to disk too
            with open('monitor-renderer.html', 'w') as fp:
                print(html, file=fp)

    @task_submitter(task_type='simulate', n_slots=1)
    def submit_calc(self):
        """Submit a calculation when resources are available"""

        # Wait if the task list is being updated
        self.task_list_ready.wait()

        with self.priority_list_lock:
            next_mol = self.priority_list.pop()  # Get the next best molecule
            self.already_ran.add(next_mol)  # Used to make sure we don't run things twice

        # Send it to the task server to run
        self.queues.send_inputs(next_mol, method='compute_vertical', topic='simulate')
        self.rec_progbar.update(1)

        # Update the allocation from the dashboard
        self._update_dashboard('simulate', 1)

    @result_processor(topic='simulate')
    def receive_calc(self, result: Result):
        """Store the output of simulation if it is successful"""

        # Store the result if successful
        if result.success:
            # Store the result in a database
            self.database[result.args[0]] = result.value

            # Mark that we've received a result
            self.sent_progbar.update(1)

            # If we've got all simulations complete, stop
            if len(self.database) >= self.n_to_evaluate:
                self.logger.info(f'Completed as many as required.')
                self.done.set()

            # Start training if enough data
            if len(self.database) >= self.initial_count and len(self.database) % self.batch_size == 0:
                self.task_list_ready.clear()  # Blocks new simulations from starting
                self.start_update.set()  # Tell the training agents to start

        # Store the result object for later processing
        self.simulation_results.append(result)

        # Mark that the resources are now free
        self._update_dashboard('simulate', -1)
        self.rec.release('simulate', 1)

    @event_responder(event_name='start_update')
    def start_training(self):
        """Start the training tasks"""

        # Start a training task with the current database
        print(f'Starting training. Database size: {len(self.database)}...', end='')
        smiles, ie = zip(*self.database.items())
        self.queues.send_inputs(smiles, ie, method='train_model', topic='train')

        # Update the allocation from the dashboard
        self._update_dashboard('train', 1)

    @result_processor(topic='train')
    def receive_new_model(self, result: Result):
        """Receive a finished model training and start inference tasks"""

        # Get the model
        assert result.success, f'Model training failed! {result.failure_info.exception}'
        model = result.value
        print('Training complete...', end='')

        # Update the allocation from the dashboard
        self._update_dashboard('train', -1)

        # Launch the inference tasks
        inf_chunks = np.array_split(self.priority_list, self.inference_tasks)
        for chunk in inf_chunks:
            self.queues.send_inputs(model, chunk, method='run_model', topic='infer')
            self._update_dashboard('infer', 1)

        # Store the output
        self.learning_results.append(result)

    @event_responder(event_name='start_update')
    def collect_inference(self):
        """Collect the inference tasks, use them to update task queue"""

        start_time = perf_counter()

        # Collect all inference tasks
        chunks = []
        for i in range(self.inference_tasks):
            result = self.queues.get_result(topic='infer')
            self._update_dashboard('infer', -1)
            assert result.success, f'Inference failed! {result.failure_info.exception}'
            chunks.append(result.value)
            self.learning_results.append(result)

        # When all are finished, sort the list ascending (best last)
        results = pd.concat(chunks, ignore_index=True).sort_values('ie', ascending=True)

        # Add them to the task queue
        with self.priority_list_lock:
            self.priority_list.clear()  # Get rid of the old list
            for smiles in results['smiles']:
                if smiles not in self.already_ran:
                    self.priority_list.append(smiles)

        # Mark that we can resume simulations
        self.task_list_ready.set()
        print(f'Inference done. Elapsed time: {perf_counter() - start_time:.2f}s', end='\n')
