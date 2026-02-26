import os, json, sys, time, re, math, random, datetime, argparse, requests
from typing import *
from dataclasses import dataclass, field


class GeometryElement:
    pass

class Point(GeometryElement):
    def __init__(self, label: Optional[str] = None):
        self.label = label
    
    def __repr__(self):
        return f"Point({self.label})"

class Line(GeometryElement):
    def __init__(self, points: Optional[List[Point]] = None):
        self.points: List[Point] = [] if points is None else points
    
    def __repr__(self):
        return f"Line({''.join((point.label or "") for point in self.points)})"

class Circle(GeometryElement):
    def __init__(self, center: Optional[Point] = None, points: Optional[List[Point]] = None):
        self.center = center
        self.points: List[Point] = [] if points is None else points
    
    def __repr__(self):
        return f"Circle(center={self.center.label if self.center else 'None'}, through={''.join((point.label or "") for point in self.points)})"

class Constraint:
    pass

class ConstraintEqual(Constraint):
    def __init__(self, dist1: Tuple[Point, Point], dist2: Tuple[Point, Point]):
        self.dist1 = dist1
        self.dist2 = dist2
        
    def __repr__(self):
        return f"ConstraintEqual({self.dist1}, {self.dist2})"

class ConstraintParallel(Constraint):
    def __init__(self, line1: Line, line2: Line):
        self.line1 = line1
        self.line2 = line2
    
    def __repr__(self):
        return f"ConstraintParallel({self.line1}, {self.line2})"

class ConstraintPerpendicular(Constraint):
    def __init__(self, line1: Line, line2: Line):
        self.line1 = line1
        self.line2 = line2
    
    def __repr__(self):
        return f"ConstraintPerpendicular({self.line1}, {self.line2})"

class ConstraintLineCircleTangent(Constraint):
    def __init__(self, line: Line, circle: Circle):
        self.line = line
        self.circle = circle
        
    def __repr__(self):
        return f"ConstraintLineCircleTangent({self.line}, {self.circle})"

class ConstraintCircleCircleTangent(Constraint):
    def __init__(self, circle1: Circle, circle2: Circle):
        self.circle1 = circle1
        self.circle2 = circle2
    
    def __repr__(self):
        return f"ConstraintCircleCircleTangent({self.circle1}, {self.circle2})"

@dataclass
class GeometryConstruction:
    points: List[Point] = field(default_factory=list)
    lines: List[Line] = field(default_factory=list)
    circles: List[Circle] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    
    # post init
    def __post_init__(self):
        self.point_available_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        random.shuffle(self.point_available_labels)
        
    def point_on_lines(self, point: Point) -> List[Line]:
        lines = []
        for line in self.lines:
            if point in line.points:
                lines.append(line)
        return lines

    def point_on_circles(self, point: Point) -> List[Circle]:
        circles = []
        for circle in self.circles:
            if point in circle.points:
                circles.append(circle)
        return circles
    
    def remove(self, shapes: GeometryElement):
        raise NotImplementedError("remove method not implemented")
    
    def add(self, shapes: Union[GeometryElement, Iterable[GeometryElement]]):
        if isinstance(shapes, GeometryElement):
            shapes = [shapes]
        for shape in shapes:
            if isinstance(shape, Point):
                if shape not in self.points:
                    self.points.append(shape)
            if isinstance(shape, Line):
                if shape not in self.lines:
                    self.lines.append(shape)
                for point in shape.points:
                    if point not in self.points:
                        self.points.append(point)
            if isinstance(shape, Circle):
                if shape not in self.circles:
                    self.circles.append(shape)
                for point in shape.points:
                    if point not in self.points:
                        self.points.append(point)
                if shape.center and shape.center not in self.points:
                    self.points.append(shape.center)
    
    def add_constraint(self, constraint: Constraint):
        if isinstance(constraint, ConstraintEqual):
            self.constraints.append(constraint)
        elif isinstance(constraint, ConstraintParallel):
            self.constraints.append(constraint)
        elif isinstance(constraint, ConstraintPerpendicular):
            self.constraints.append(constraint)
        elif isinstance(constraint, ConstraintLineCircleTangent):
            self.constraints.append(constraint)
        elif isinstance(constraint, ConstraintCircleCircleTangent):
            self.constraints.append(constraint)
        else:
            raise ValueError("Unknown constraint type")
        
    def pop_point_label(self) -> str:
        return self.point_available_labels.pop()
            
    def get_triangles(self) -> List[Tuple[Point, Point, Point, Line, Line, Line]]:
        triangles = []
        for i in range(len(self.points)):
            for j in range(i + 1, len(self.points)):
                for k in range(j + 1, len(self.points)):
                    i_on_lines = set(self.point_on_lines(self.points[i]))
                    j_on_lines = set(self.point_on_lines(self.points[j]))
                    k_on_lines = set(self.point_on_lines(self.points[k]))
                    ij_on_lines = i_on_lines.intersection(j_on_lines)
                    ik_on_lines = i_on_lines.intersection(k_on_lines)
                    jk_on_lines = j_on_lines.intersection(k_on_lines)
                    ijk_on_lines = ij_on_lines.intersection(k_on_lines)
                    if len(ij_on_lines) == 0 or len(ik_on_lines) == 0 or len(jk_on_lines) == 0 or len(ijk_on_lines) > 0:
                        continue
                    ij_line = random.choice(list(ij_on_lines))
                    ik_line = random.choice(list(ik_on_lines))
                    jk_line = random.choice(list(jk_on_lines))
                    triangles.append((self.points[i], self.points[j], self.points[k], jk_line, ik_line, ij_line))
        return triangles

    def json(self):
        point_to_index = {point: i for i, point in enumerate(self.points)}
        line_to_index = {line: i for i, line in enumerate(self.lines)}
        circle_to_index = {circle: i for i, circle in enumerate(self.circles)}
        data = {
            "points": [point.label for point in self.points],
            "lines": [{"points": [point_to_index[point] for point in line.points]} for line in self.lines],
            "circles": [{"center": point_to_index[circle.center] if circle.center else None, "points": [point_to_index[point] for point in circle.points]} for circle in self.circles],
            "constraints": []
        }
        for constraint in self.constraints:
            if isinstance(constraint, ConstraintEqual):
                data["constraints"].append({
                    "type": "ConstraintEqual",
                    "dist1": [point_to_index[constraint.dist1[0]], point_to_index[constraint.dist1[1]]],
                    "dist2": [point_to_index[constraint.dist2[0]], point_to_index[constraint.dist2[1]]]
                })
            elif isinstance(constraint, ConstraintParallel):
                data["constraints"].append({
                    "type": "ConstraintParallel",
                    "line1": line_to_index[constraint.line1],
                    "line2": line_to_index[constraint.line2]
                })
            elif isinstance(constraint, ConstraintPerpendicular):
                data["constraints"].append({
                    "type": "ConstraintPerpendicular",
                    "line1": line_to_index[constraint.line1],
                    "line2": line_to_index[constraint.line2]
                })
            elif isinstance(constraint, ConstraintLineCircleTangent):
                data["constraints"].append({
                    "type": "ConstraintLineCircleTangent",
                    "line": line_to_index[constraint.line],
                    "circle": circle_to_index[constraint.circle]
                })
            elif isinstance(constraint, ConstraintCircleCircleTangent):
                data["constraints"].append({
                    "type": "ConstraintCircleCircleTangent",
                    "circle1": circle_to_index[constraint.circle1],
                    "circle2": circle_to_index[constraint.circle2]
                })
            else:
                raise ValueError("Unknown constraint type")
        return data
    
    @staticmethod
    def from_json(data: dict):
        shape = GeometryConstruction()
        for point_label in data["points"]:
            shape.points.append(Point(point_label))
            if point_label is not None:
                shape.point_available_labels.remove(point_label)
        for line_points in data["lines"]:
            line = Line()
            for point_idx in line_points["points"]:
                point = shape.points[point_idx]
                line.points.append(point)
            shape.lines.append(line)
        for circle_points in data["circles"]:
            circle = Circle()
            for point_idx in circle_points["points"]:
                point = shape.points[point_idx]
                circle.points.append(point)
            if circle_points["center"]:
                center = shape.points[circle_points["center"]]
                circle.center = center
            shape.circles.append(circle)
        for constraint_data in data["constraints"]:
            if constraint_data["type"] == "ConstraintEqual":
                dist1 = (shape.points[constraint_data["dist1"][0]], shape.points[constraint_data["dist1"][1]])
                dist2 = (shape.points[constraint_data["dist2"][0]], shape.points[constraint_data["dist2"][1]])
                shape.add_constraint(ConstraintEqual(dist1, dist2))
            elif constraint_data["type"] == "ConstraintParallel":
                line1 = shape.lines[constraint_data["line1"]]
                line2 = shape.lines[constraint_data["line2"]]
                shape.add_constraint(ConstraintParallel(line1, line2))
            elif constraint_data["type"] == "ConstraintPerpendicular":
                line1 = shape.lines[constraint_data["line1"]]
                line2 = shape.lines[constraint_data["line2"]]
                shape.add_constraint(ConstraintPerpendicular(line1, line2))
            elif constraint_data["type"] == "ConstraintLineCircleTangent":
                line = shape.lines[constraint_data["line"]]
                circle = shape.circles[constraint_data["circle"]]
                shape.add_constraint(ConstraintLineCircleTangent(line, circle))
            elif constraint_data["type"] == "ConstraintCircleCircleTangent":
                circle1 = shape.circles[constraint_data["circle1"]]
                circle2 = shape.circles[constraint_data["circle2"]]
                shape.add_constraint(ConstraintCircleCircleTangent(circle1, circle2))
            else:
                raise ValueError("Unknown constraint type")
        return shape

