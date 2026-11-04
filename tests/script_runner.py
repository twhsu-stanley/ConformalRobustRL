"""Exercise executable script bodies with small test configurations."""

import ast
import importlib
from pathlib import Path


def run_script(module_name, **settings):
    """Override initial settings in a script's main guard without running its large defaults."""
    module = importlib.import_module(module_name)
    path = Path(module.__spec__.origin)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    guard = next(node for node in tree.body if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "__name__ == '__main__'")
    remaining = set(settings)
    for node in guard.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in remaining:
                node.value = ast.Subscript(ast.Name("_script_settings", ast.Load()),
                                           ast.Constant(target.id), ast.Load())
                remaining.remove(target.id)
    if remaining:
        raise ValueError(f"Unknown script settings: {sorted(remaining)}")
    namespace = dict(vars(module), _script_settings=settings)
    body = ast.fix_missing_locations(ast.Module(body=guard.body, type_ignores=[]))
    exec(compile(body, str(path), "exec"), namespace)
    return namespace


def script_entrypoint(module_name, result, *, positional=(), **defaults):
    """Adapt existing integration cases to the executable scripts, using one run per case."""
    stem = module_name.rsplit(".", 1)[-1]

    def run(*args, **settings):
        if len(args) > len(positional):
            raise TypeError("Too many positional test settings.")
        settings = defaults | dict(zip(positional, args)) | settings
        if stem.endswith("_online"):
            settings = dict(n_deployment_episodes=30, eta=0.01, pilot_episodes=20,
                            R_star=0.2, save_plots=True) | settings
            settings["seeds"] = (settings.pop("seed", 7),)
            settings["methods"] = (settings.pop("method", "proposed"),)
            settings["first_trial"] = settings.pop("trial", 1)
        elif stem.startswith("main_robot_"):
            settings["experiments"] = ((settings.pop("R", 0.15), settings.pop("C", 1.2),
                                        settings.pop("q_initializer", None),
                                        settings.pop("n_trials", 15)),)
            settings["checkpoint_path"] = None
        elif stem == "plot_fixedRC_value_policy":
            settings["combinations"] = ((settings.pop("R", 0.2), settings.pop("C", 1.0)),)
            settings.setdefault("show", False)
        elif stem == "train_robot_3d_transformer":
            settings = dict(learning_rate=2e-5, head_learning_rate=1e-4,
                            position_learning_rate=5e-5, resume_from_checkpoint=None,
                            restart_lr_schedule=False) | settings
        for name in ("output_dir", "data_dir", "dataset_path", "checkpoint_path"):
            if settings.get(name) is not None:
                settings[name] = Path(settings[name])
        return run_script(module_name, **settings)[result]

    return run
