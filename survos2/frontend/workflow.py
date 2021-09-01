import yaml
import os
from loguru import logger

from survos2.frontend.control.launcher import Launcher
from survos2.model import DataModel
from survos2.server.state import cfg

from napari.qt.threading import thread_worker
from napari.qt import progress


@thread_worker
def run_workflow(msg):
    workflow_file = msg["workflow_file"]

    if not os.path.isabs(workflow_file):
        workflow_file = os.path.join(os.getcwd(), workflow_file)

    with open(workflow_file) as f:
        workflows = yaml.safe_load(f.read())

    num_workflow_steps = len(workflows.keys())
    minVal, maxVal = 0, num_workflow_steps
    # with ProgressDialog(
    #    f"Processing pipeline {workflow_file}", minVal, maxVal
    # ) as dlg:
    with progress(total=num_workflow_steps) as pbar:
        # if dlg.wasCanceled():
        #    raise Exception("Processing canceled")

        for step_idx, k in enumerate(workflows):
            workflow = workflows[k]
            action = workflow.pop("action")
            plugin, command = action.split(".")
            params = workflow.pop("params")

            src_name = workflow.pop("src")
            dst_name = workflow.pop("dst")

            if "src_group" in workflow:
                src_group = workflow.pop("src_group")
                src = DataModel.g.dataset_uri(src_name, group=src_group)
            else:
                src = DataModel.g.dataset_uri(src_name, group=plugin)

            dst = DataModel.g.dataset_uri(dst_name, group=plugin)

            all_params = dict(src=src, dst=dst, modal=True)
            all_params.update(params)
            logger.info(f"Executing workflow {all_params}")

            logger.debug(
                f"+ Running {plugin}, {command} on {src}\n to dst {dst} {all_params}\n"
            )

            Launcher.g.run(plugin, command, **all_params)
            # dlg.setValue(step_idx)
            pbar.update(1)
    cfg.ppw.clientEvent.emit(
        {"source": "workspace_gui", "data": "refresh", "value": None}
    )
