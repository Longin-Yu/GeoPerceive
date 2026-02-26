from src import solver, definition, generator
import os, json, sys, time, re, math, random, datetime, argparse, requests, multiprocessing
from typing import *
from tqdm import tqdm

def solve(shape: definition.GeometryConstruction, save_path_prefix: str):
    """
    Solve the shape instance and save the result to the specified path.
    """
    # Create a solver instance
    s = solver.GeometrySolver()
    generator_point_to_solver: Dict[definition.Point, solver.Point] = {}
    generator_line_to_solver: Dict[definition.Line, solver.Line] = {}
    generator_circle_to_solver: Dict[definition.Circle, solver.Circle] = {}
    for point in shape.points:
        generator_point_to_solver[point] = solver.Point(label=point.label)
        s.add_object(generator_point_to_solver[point])
    for line in shape.lines:
        generator_line_to_solver[line] = solver.Line(
            through=list(map(lambda x: generator_point_to_solver[x], line.points)),
        )
        s.add_object(generator_line_to_solver[line])
    for circle in shape.circles:
        generator_circle_to_solver[circle] = solver.Circle(
            center=generator_point_to_solver[circle.center] if circle.center else None,
            through=list(map(lambda x: generator_point_to_solver[x], circle.points)),
        )
        s.add_object(generator_circle_to_solver[circle])
    for constraint in shape.constraints:
        if isinstance(constraint, definition.ConstraintEqual):
            s.register_loss(
                f"external/equal.{constraint.dist1}={constraint.dist2}",
                solver.constraint_equal(
                    tuple(map(lambda x: generator_point_to_solver[x], constraint.dist1)),
                    tuple(map(lambda x: generator_point_to_solver[x], constraint.dist2)),
                )
            )
        elif isinstance(constraint, definition.ConstraintPerpendicular):
            s.register_loss(
                f"external/perpendicular.{constraint.line1.__hash__()}={constraint.line2.__hash__()}",
                solver.constraint_perpendicular(
                    generator_line_to_solver[constraint.line1],
                    generator_line_to_solver[constraint.line2],
                )
            )
        elif isinstance(constraint, definition.ConstraintParallel):
            s.register_loss(
                f"external/parallel.{constraint.line1.__hash__()}={constraint.line2.__hash__()}",
                solver.constraint_parallel(
                    generator_line_to_solver[constraint.line1],
                    generator_line_to_solver[constraint.line2],
                )
            )
        elif isinstance(constraint, definition.ConstraintLineCircleTangent):
            s.register_loss(
                f"external/lctangent.{constraint.line.__hash__()}={constraint.circle.__hash__()}",
                solver.constraint_line_circle_tangent(
                    generator_line_to_solver[constraint.line],
                    generator_circle_to_solver[constraint.circle],
                )
            )
        else:
            raise NotImplementedError(f"Unknown constraint type: {type(constraint)}")
    s.register_internal_loss(weight=1)
    s.register_point_density_penalty(weight=0.2, threshold=0.2)
    s.register_point_distribution_penalty(weight=0.1, threshold=1.0)
    
    final_loss_tree = s.solve(iterations=10000, lr=0.1, threshold=3e-5, step_size=50, gamma=0.965, verbose=False)
    
    fig_save_path = save_path_prefix + ".jpg"
    loss_save_path = save_path_prefix + ".json"
    solver.plot_solution(s.objects, fig_save_path)
    with open(loss_save_path, 'w') as f:
        json.dump(final_loss_tree, f, indent=4)
    if final_loss_tree["constraints"]["loss"] < 3e-5:
        return True
    else:
        print(f"Failed to solve shape: {shape}")
        return False
    
def solve_from_json(index_and_json):
    index, json_str, save_dir = index_and_json
    data = json.loads(json_str)
    save_path_prefix = os.path.join(save_dir, 'images', f'{index}')
    if os.path.exists(save_path_prefix + ".json") and os.path.exists(save_path_prefix + ".jpg"):
        return (index, True)
    shape = definition.GeometryConstruction.from_json(data['data'])
    success = solve(shape, save_path_prefix)
    return (index, success)

def set_all_seed(seed: int):
    """
    Set the random seed for all libraries.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    arg_parser = argparse.ArgumentParser(description='Generate data for the solver')
    arg_parser.add_argument('--save_dir', type=str, default='data', help='Directory to save generated data')
    arg_parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate')
    arg_parser.add_argument('--min_steps', type=int, default=1, help='Minimum number of steps')
    arg_parser.add_argument('--max_steps', type=int, default=10, help='Maximum number of steps')
    arg_parser.add_argument('--num_processes', type=int, default=4, help='Number of processes to use for solving')
    arg_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    arg_parser.add_argument('--no_solve', action='store_true', help='Skip solving and only generate data')
    arg_parser.add_argument('--resume', action='store_true', help='Resume from the last saved sample')
    args = arg_parser.parse_args()
    
    meta_file_path = os.path.join(args.save_dir, 'meta.jsonl')
    
    set_all_seed(args.seed)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    generated_indices = set()
    
    if args.resume:
        if not os.path.exists(meta_file_path):
            raise FileNotFoundError(f"Meta file {meta_file_path} does not exist. Cannot resume.")
        with open(meta_file_path, 'r') as meta_file:
            lines = meta_file.readlines()
        for line in lines:
            piece = json.loads(line)
            generated_indices.add(piece["index"])
        print(f"Found {len(generated_indices)} items in meta file.")
    else:
        if os.path.exists(meta_file_path):
            print(f"Meta file {meta_file_path} already exists. Overwriting.")
            os.remove(meta_file_path)
            
    print(f"\033[96mGenerating {args.num_samples} samples with steps between {args.min_steps} and {args.max_steps}\033[0m")
    
    with open(meta_file_path, 'a') as meta_file:
        for i in tqdm(range(args.num_samples)):
            if i in generated_indices:
                continue
            steps = random.randint(args.min_steps, args.max_steps)
            sample = definition.GeometryConstruction()
            g = generator.ConstructionGenerator(sample)
            constructions = g.generate(steps)
            data = {
                "index": i,
                "step": steps,
                "data": sample.json(),
                "constructions": constructions,
            }
            meta_file.write(json.dumps(data) + '\n')
    
    if args.no_solve:
        print(f"\033[92mData generation completed. Skipping solving.\033[0m")
        return
    

    diagram_dir = os.path.join(args.save_dir, 'images')
    if not os.path.exists(diagram_dir):
        os.makedirs(diagram_dir)

    print(f"\033[96mSolving and drawing diagrams for {args.num_samples} samples using {args.num_processes} processes\033[0m")

    with open(meta_file_path, 'r') as meta_file:
        lines = meta_file.readlines()

    inputs = [(i, line, args.save_dir) for i, line in enumerate(lines)]
    manager = multiprocessing.Manager()
    lock = manager.Lock()            # 避免同时写入导致冲突
    pbar = tqdm(total=len(inputs), desc="Solving")
    
    def update_progress(*_):
        with lock:
            pbar.update(1)

    with multiprocessing.Pool(processes=args.num_processes) as pool:
        results_async = [
            pool.apply_async(solve_from_json, args=(inp,), callback=update_progress) for inp in inputs
        ]
        results = [r.get() for r in results_async]

    failed = [i for i, success in results if not success]
    if failed:
        print(f"\033[91mFailed to solve {len(failed)} samples: {failed}\033[0m")
    else:
        print(f"\033[92mAll samples solved successfully!\033[0m")

if __name__ == '__main__':
    main()