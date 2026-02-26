import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class Point:
    def __init__(self, label=None):
        self.label = label
        self.coord = torch.nn.Parameter(torch.rand(2), requires_grad=True)
    
    def internal_loss(self):
        return torch.tensor(0.0, dtype=torch.float32)
    
    def __repr__(self):
        return f"Point({self.label}: {self.coord.data.tolist()})"

class Line:
    def __init__(self, label=None, through=None):
        self.label = label
        self.through = through if through is not None else []
        self.a = torch.nn.Parameter(torch.randn(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.randn(1), requires_grad=True)
        self.c = torch.nn.Parameter(torch.randn(1), requires_grad=True)
    
    def internal_loss(self):
        loss = torch.tensor(0.0, dtype=torch.float32)
        norm = torch.sqrt(self.a**2 + self.b**2 + 1e-8)
        for p in self.through:
            val = self.a * p.coord[0] + self.b * p.coord[1] + self.c
            loss = loss + (val / norm)**2
        norm_params = (torch.sqrt(self.a ** 2 + self.b ** 2 + self.c ** 2) - 1) ** 2
        return loss + norm_params
    
    def get_xrange(self):
        if len(self.through) == 0:
            return None
        xmin, xmax = torch.inf, -torch.inf
        for point in self.through:
            xmin = min(xmin, point.coord[0].item())
            xmax = max(xmax, point.coord[0].item())
        return xmin, xmax
    
    def get_yrange(self):
        if len(self.through) == 0:
            return None
        ymin, ymax = torch.inf, -torch.inf
        for point in self.through:
            ymin = min(ymin, point.coord[1].item())
            ymax = max(ymax, point.coord[1].item())
        return ymin, ymax
    
    def __repr__(self):
        return f"Line({self.label}: a={self.a.item():.3f}, b={self.b.item():.3f}, c={self.c.item():.3f})"

class Circle:
    def __init__(self, label=None, center=None, through=None):
        self.label = label
        self.through = through if through is not None else []
        self.center_ref = center
        self.center = torch.nn.Parameter(torch.rand(2), requires_grad=True)
        self.radius = torch.nn.Parameter(torch.rand(1), requires_grad=True)
    
    def internal_loss(self):
        loss = torch.tensor(0.0, dtype=torch.float32)
        if self.center_ref:
            loss += torch.norm(self.center - self.center_ref.coord)
        for p in self.through:
            dist = torch.norm(p.coord - self.center)
            loss = loss + ((dist - self.radius).squeeze())**2
        return loss
    
    def __repr__(self):
        center_val = self.center.data.tolist() if isinstance(self.center, torch.Tensor) else self.center.tolist()
        return f"Circle({self.label}: center={center_val}, radius={self.radius.data.tolist()})"

def constraint_equal(seg1, seg2):
    def loss_fn():
        P1, P2 = seg1
        P3, P4 = seg2
        diff1 = torch.norm(P1.coord - P2.coord)
        diff2 = torch.norm(P3.coord - P4.coord)
        return (diff1 - diff2)**2
    return loss_fn

def constraint_perpendicular(line1, line2):
    def loss_fn():
        a1, b1 = line1.a, line1.b
        a2, b2 = line2.a, line2.b
        return (a1 * a2 + b1 * b2)**2
    return loss_fn

def constraint_parallel(line1, line2):
    def loss_fn():
        a1 = line1.a
        b1 = line1.b
        a2 = line2.a
        b2 = line2.b
        cross = a1 * b2 - a2 * b1
        return cross**2
    return loss_fn

def constraint_line_circle_tangent(line, circle):
    def loss_fn():
        a = line.a
        b = line.b
        c = line.c
        norm = torch.sqrt(a**2 + b**2 + 1e-8)
        center = circle.center
        dist = torch.abs(a * center[0] + b * center[1] + c) / norm
        return (dist - circle.radius.squeeze())**2
    return loss_fn

def constraint_assign_length(segment, target_length):
    def loss_fn():
        P1, P2 = segment
        diff = torch.norm(P1.coord - P2.coord)
        return (diff - target_length)**2
    return loss_fn

def constraint_assign_angle(vertex, p1, p2, target_angle, radians=False):
    def loss_fn():
        v1 = p1.coord - vertex.coord
        v2 = p2.coord - vertex.coord
        dot = torch.dot(v1, v2)
        norm_product = torch.norm(v1) * torch.norm(v2) + 1e-8
        current_angle = torch.acos(dot / norm_product)
        target = target_angle if radians else target_angle * torch.pi / 180.0
        return (current_angle - target)**2
    return loss_fn

def penalty_point_distribution(points, threshold):
    def loss_fn():
        if not points:
            return torch.tensor(0.0, dtype=torch.float32)
        coords = torch.stack([p.coord for p in points])
        centroid = torch.mean(coords, dim=0)
        distances = torch.norm(coords - centroid, dim=1)
        loss = torch.tensor(0.0, dtype=torch.float32)
        for d in distances:
            excess = torch.relu(d - threshold)
            loss = loss + excess**2
        return loss
    return loss_fn


class GeometrySolver:
    def __init__(self):
        self.objects = []
        self.constraints = {}
        self.penalties = {}
        
    def add_object(self, obj):
        self.objects.append(obj)
        
    def all_points(self):
        pts = []
        for obj in self.objects:
            if isinstance(obj, Point):
                pts.append(obj)
            elif isinstance(obj, Line):
                pts.extend(obj.through)
            elif isinstance(obj, Circle):
                pts.extend(obj.through)
        # 保持唯一性
        unique_pts = {id(p): p for p in pts}
        return list(unique_pts.values())
    
    def register_loss(self, key: str, loss_fn, weight: float = 1, loss_type: str = "constraint"):
        tree = self.penalties if loss_type == "penalty" else self.constraints
        parts = key.split("/")
        current = tree
        for part in parts[:-1]:
            if part not in current:
                current[part] = (1.0, {})
            else:
                if not isinstance(current[part][1], dict):
                    current[part] = (current[part][0], {})
            current = current[part][1]
        current[parts[-1]] = (weight, loss_fn)
    
    def register_internal_loss(self, weight: float):
        for i, obj in enumerate(self.objects):
            if isinstance(obj, Point):
                continue 
            label = obj.label if obj.label is not None else f"obj_{i}"
            key = f"internal/{label}"
            self.register_loss(key, obj.internal_loss, weight, loss_type="constraint")
    
    def register_point_density_penalty(self, weight: float, threshold: float):
        pts = self.all_points()
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                key = f"density/point_{i}_{j}"
                # 使用默认参数绑定防止闭包问题
                def loss_fn(pt1=pts[i], pt2=pts[j], threshold=threshold):
                    d = torch.norm(pt1.coord - pt2.coord)
                    if d.item() < threshold:
                        return (1.0/((d**2)+1e-8) - 1.0/(threshold**2))
                    else:
                        return torch.tensor(0.0, dtype=torch.float32)
                self.register_loss(key, loss_fn, weight, loss_type="penalty")
    
    def register_point_distribution_penalty(self, weight: float, threshold: float):
        pts = self.all_points()
        key = "penalty/point_distribution"
        self.register_loss(key, penalty_point_distribution(pts, threshold), weight, loss_type="penalty")
    
    def evaluate_tree_losses(self, tree):
        total = 0.0
        total_weight = 0.0
        for key, (w, value) in tree.items():
            total_weight += w
            if callable(value):
                total = total + w * value()
            elif isinstance(value, dict):
                total = total + w * self.evaluate_tree_losses(value)
        return total / total_weight if total_weight != 0 else total
    
    def get_loss_tree(self, tree):
        children = {}
        total_loss = 0.0
        total_weight = 0.0
        for key, (w, value) in tree.items():
            total_weight += w
            if callable(value):
                loss_val = value().item()
                total_loss += w * loss_val
                children[key] = {"weight": w, "loss": loss_val}
            elif isinstance(value, dict):
                subtree = self.get_loss_tree(value)
                children[key] = subtree
                children[key]["weight"] = w
                total_loss += w * subtree["loss"]
        return {"loss": total_loss / total_weight if total_weight != 0 else total_loss, "children": children}
    
    def get_all_constraint_leaves(self, tree=None):
        if tree is None:
            tree = self.constraints
        leaves = []
        for key, (w, value) in tree.items():
            if callable(value):
                leaves.append((w, value().item()))
            elif isinstance(value, dict):
                leaves.extend(self.get_all_constraint_leaves(value))
        return leaves
    
    def solve(self, iterations=1000, lr=0.01, threshold=None, penalty_weight=0.1, step_size=50, gamma=0.95, verbose=True):
        params = []
        for obj in self.objects:
            if isinstance(obj, Point):
                params.append(obj.coord)
            elif isinstance(obj, Line):
                params.extend([obj.a, obj.b, obj.c])
            elif isinstance(obj, Circle):
                params.append(obj.center)
                params.append(obj.radius)
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
        
        with tqdm(range(iterations), desc="Solving", disable=not verbose) as pbar:
            for i in pbar:
                optimizer.zero_grad()
                constraint_loss = self.evaluate_tree_losses(self.constraints)
                penalty_loss = self.evaluate_tree_losses(self.penalties)
                loss = constraint_loss + penalty_loss * penalty_weight
                pbar.set_postfix(
                    loss=loss.item(), 
                    constraint_loss=constraint_loss.item(),
                    penalty_loss=penalty_loss.item(),
                    lr=optimizer.param_groups[0]["lr"]
                )
                # 当每个约束项（忽略惩罚项）均低于阈值时提前退出
                if threshold is not None:
                    leaves = self.get_all_constraint_leaves()
                    if all(l < threshold for w, l in leaves):
                        if verbose:
                            print(f"Exit at {i}-th step, constraint loss terms: {[w*l for w, l in leaves]}")
                        break
                loss.backward()
                optimizer.step()
                scheduler.step()
        return {"constraints": self.get_loss_tree(self.constraints), "penalties": self.get_loss_tree(self.penalties)}


def plot_solution(objects, save_path: str):
    fig, ax = plt.subplots()
    xs, ys = [], []
    for obj in objects:
        if isinstance(obj, Point):
            coord = obj.coord.detach().numpy()
            xs.append(coord[0])
            ys.append(coord[1])
            ax.plot(coord[0], coord[1], 'bo')
            if obj.label:
                ax.text(coord[0], coord[1], f" {obj.label}", fontsize=12)
    if xs and ys:
        xmin, xmax = min(xs)-1, max(xs)+1
        ymin, ymax = min(ys)-1, max(ys)+1
    else:
        xmin, xmax, ymin, ymax = -10, 10, -10, 10
    for obj in objects:
        if isinstance(obj, Line):
            a = obj.a.detach().item()
            b = obj.b.detach().item()
            c = obj.c.detach().item()
            norm = np.sqrt(a*a + b*b)
            norm = norm if norm > 1e-8 else 1e-8
            a_norm, b_norm, c_norm = a/norm, b/norm, c/norm
            if len(obj.through) < 2:
                x_range = (xmin, xmax)
                y_range = (ymin, ymax)
            else:
                x_range = obj.get_xrange()
                y_range = obj.get_yrange()
            if abs(b_norm) > 1e-6:
                x_vals = np.linspace(x_range[0], x_range[1], 100)
                y_vals = (-a_norm * x_vals - c_norm) / b_norm
                ax.plot(x_vals, y_vals, 'r-', label=obj.label if obj.label else "")
            else:
                x_val = -c_norm / a_norm
                y_vals = np.linspace(y_range[0], y_range[1], 100)
                ax.plot(np.full_like(y_vals, x_val), y_vals, 'r-', label=obj.label if obj.label else "")
        elif isinstance(obj, Circle):
            center = obj.center.detach().numpy()
            radius = obj.radius.detach().item()
            circle_patch = plt.Circle((center[0], center[1]), radius, color='g', fill=False,
                                      label=obj.label if obj.label else "")
            ax.add_patch(circle_patch)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')
    # ax.
    # plt.figure(figsize=(8, 8))
    plt.axis('off')
    # plt.legend()
    # plt.show()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)


def print_loss_tree(loss_tree, indent=0):
    for key, node in loss_tree.items():
        prefix = " " * indent
        if "children" in node:
            print(f"{prefix}{key}: weight={node.get('weight', None)}, loss={node['loss']:.6f}")
            print_loss_tree(node["children"], indent=indent+4)
        else:
            print(f"{prefix}{key}: weight={node['weight']}, loss={node['loss']:.6f}")

if __name__ == "__main__":
    solver = GeometrySolver()
    
    A = Point(label="A")
    B = Point(label="B")
    C = Point(label="C")
    H = Point(label="H")
    # M = Point(label="M")
    # N = Point(label="N")
    solver.add_object(A)
    solver.add_object(B)
    solver.add_object(C)
    solver.add_object(H)
    # solver.add_object(M)
    # solver.add_object(N)
    
    line_AB = Line(label="AB", through=[A, B])
    line_BC = Line(label="BC", through=[B, C])
    line_CA = Line(label="CA", through=[C, A])
    # line_AH = Line(label="AH", through=[A, H, M])
    # line_BH = Line(label="BH", through=[B, H, N])
    line_AH = Line(label="AH", through=[A, H])
    line_BH = Line(label="BH", through=[B, H])
    line_CH = Line(label="CH", through=[C, H])
    # line_MN = Line(label="MN", through=[M, N])
    solver.add_object(line_AB)
    solver.add_object(line_BC)
    solver.add_object(line_CA)
    solver.add_object(line_AH)
    solver.add_object(line_BH)
    solver.add_object(line_CH)
    # solver.add_object(line_MN)
    
    circle_gamma = Circle(center=H)
    solver.add_object(circle_gamma)
    
    solver.register_loss("external/vertical.line_AB_line_CH", constraint_perpendicular(line_AB, line_CH), 1.0, loss_type="constraint")
    solver.register_loss("external/vertical.line_BC_line_AH", constraint_perpendicular(line_BC, line_AH), 1.0, loss_type="constraint")
    solver.register_loss("external/vertical.line_CA_line_BH", constraint_perpendicular(line_CA, line_BH), 1.0, loss_type="constraint")
    solver.register_loss("external/tangent.line_AB_circle_gamma", constraint_line_circle_tangent(line_AB, circle_gamma), 1.0, loss_type="constraint")
    # solver.register_loss("external/parallel/line_MN_line_AB", constraint_parallel(line_MN, line_AB), 1.0, loss_type="constraint")
    
    solver.register_internal_loss(weight=1)
    
    solver.register_point_density_penalty(weight=0.2, threshold=0.2)
    
    solver.register_point_distribution_penalty(weight=0.1, threshold=1.0)
    
    final_loss_tree = solver.solve(iterations=10000, lr=0.1, threshold=3e-5)
    
    print_loss_tree(final_loss_tree["constraints"]["children"])
    print_loss_tree(final_loss_tree["penalties"]["children"])
    
    for obj in solver.objects:
        print(obj)
    
    plot_solution(solver.objects, "local_demo.jpg")
