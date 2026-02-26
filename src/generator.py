import os, json, sys, time, re, math, random, datetime, argparse, requests
from typing import *
from dataclasses import dataclass, field

from src.definition import *

@dataclass
class ConstructionDescription:
    description: str
    point_map: Dict[str, Point] = field(default_factory=dict)
    line_map: Dict[str, Line] = field(default_factory=dict)
    circle_map: Dict[str, Circle] = field(default_factory=dict)
    
    def format(self) -> str:
        return self.description.format(**{**self.point_map, **self.line_map, **self.circle_map})

class ConstructionProcessor:
    def __init__(self, construction: GeometryConstruction):
        self.construction = construction
    
    def prepare(self) -> bool:
        raise NotImplementedError
    
    def apply(self) -> ConstructionDescription:
        raise NotImplementedError
    
class ConstructOrthocentre(ConstructionProcessor):
    def prepare(self):
        available_triangles = self.construction.get_triangles()
                    
        if len(available_triangles) == 0:
            return False
        if len(self.construction.point_available_labels) < 1:
            return False
        self.states = random.choice(available_triangles)
        return True
    
    def apply(self):
        A, B, C, BC, AC, AB = self.states
        h_use_label = random.random() < 0.5
        H = Point(self.construction.pop_point_label() if h_use_label else None)
        AH = Line(points=[A, H])
        BH = Line(points=[B, H])
        CH = Line(points=[C, H])
        self.construction.add([H, AH, BH, CH])
        self.construction.add_constraint(ConstraintPerpendicular(AH, BC))
        self.construction.add_constraint(ConstraintPerpendicular(BH, AC))
        self.construction.add_constraint(ConstraintPerpendicular(CH, AB))
        return ConstructionDescription(
            description="Construct the orthocentre {H} of a triangle with vertices {A}, {B}, and {C}.",
            point_map={"H": H, "A": A, "B": B, "C": C}
        )

class ConstructCircumcentre(ConstructionProcessor):
    def prepare(self):
        available_triangles = self.construction.get_triangles()
                    
        if len(available_triangles) == 0:
            return False
        if len(self.construction.point_available_labels) < 1:
            return False
        self.states = random.choice(available_triangles)
        return True
    
    def apply(self):
        A, B, C, BC, AC, AB = self.states
        create_center = random.random() < 0.5
        center = Point(self.construction.pop_point_label()) if create_center else None
        circle = Circle(center=center, points=[A, B, C])
        self.construction.add(circle)
        
        return ConstructionDescription(
            description="Construct the circumcircle {O} of a triangle with vertices {A}, {B}, and {C}." if create_center else "Construct the circumcircle of a triangle with vertices {A}, {B}, and {C}.",
            point_map={"O": center, "A": A, "B": B, "C": C} if create_center else {"A": A, "B": B, "C": C},
            circle_map={"circO": circle}
        )

class ConstructIncentre(ConstructionProcessor):
    def prepare(self):
        available_triangles = self.construction.get_triangles()
                    
        if len(available_triangles) == 0:
            return False
        if len(self.construction.point_available_labels) < 4:
            return False
        self.states = random.choice(available_triangles)
        return True
    
    def apply(self):            
        ret = ConstructionDescription(description="")
        A, B, C, BC, AC, AB = self.states
        create_center = random.random() < 0.5
        center = Point(self.construction.pop_point_label()) if create_center else None
        circle = Circle(center=center, points=[A, B, C])
        self.construction.add(circle)
        self.construction.add_constraint(ConstraintLineCircleTangent(BC, circle))
        self.construction.add_constraint(ConstraintLineCircleTangent(AC, circle))
        self.construction.add_constraint(ConstraintLineCircleTangent(AB, circle))
        ret.point_map.update({"A": A, "B": B, "C": C})
        ret.circle_map.update({"circI": circle})
        if create_center:
            ret.point_map.update({"I": center})
            ret.description += "Construct the incircle {I} of a triangle with vertices {A}, {B}, and {C}. "
        else:
            ret.description += "Construct the incircle of a triangle with vertices {A}, {B}, and {C}. "
        if random.random() < 0.4:
            point_on_BC = Point(self.construction.pop_point_label())
            circle.points.append(point_on_BC)
            BC.points.append(point_on_BC)
            self.construction.add(point_on_BC)
            ret.point_map.update({"PBC": point_on_BC})
            ret.line_map.update({"BC": BC})
            ret.description += "The incircle intersects the side {BC} at point {PBC}. "
        if random.random() < 0.4:
            point_on_AC = Point(self.construction.pop_point_label())
            circle.points.append(point_on_AC)
            AC.points.append(point_on_AC)
            self.construction.add(point_on_AC)
            ret.point_map.update({"PAC": point_on_AC})
            ret.line_map.update({"AC": AC})
            ret.description += "The incircle intersects the side {AC} at point {PAC}. "
        if random.random() < 0.4:
            point_on_AB = Point(self.construction.pop_point_label())
            circle.points.append(point_on_AB)
            AB.points.append(point_on_AB)
            self.construction.add(point_on_AB)
            ret.point_map.update({"PAB": point_on_AB})
            ret.line_map.update({"AB": AB})
            ret.description += "The incircle intersects the side {AB} at point {PAB}. "
        return ret

class ConstructSegment(ConstructionProcessor):
    def prepare(self):
        available_pairs: List[Tuple[Point, Point]] = []
        for i in range(len(self.construction.points)):
            for j in range(i + 1, len(self.construction.points)):
                i_on_lines = set(self.construction.point_on_lines(self.construction.points[i]))
                j_on_lines = set(self.construction.point_on_lines(self.construction.points[j]))
                ij_on_lines = i_on_lines.intersection(j_on_lines)
                if len(ij_on_lines) > 0:
                    continue
                available_pairs.append((self.construction.points[i], self.construction.points[j]))
        if len(available_pairs) == 0:
            return False
        self.states = random.choice(available_pairs)
        return True
    
    def apply(self):
        A, B = self.states
        segment = Line(points=[A, B])
        self.construction.add(segment)
        return ConstructionDescription(
            description=random.choice([
                "Construct a segment with endpoints {A} and {B}.",
                "Connect {A} and {B}.",
                "Link {A} and {B}.",
            ]),
            point_map={"A": A, "B": B},
            line_map={"AB": segment}
        )
    
class ConstructTwoPointsAndConnect(ConstructionProcessor):
    def prepare(self):
        construction = self.construction
        if len(construction.point_available_labels) < 2:
            return False
        
        available_pairs: List[Tuple[Union[Line, Circle], Union[Line, Circle]]] = []
        
        # Line to Line
        for i in range(len(construction.lines)):
            for j in range(i + 1, len(construction.lines)):
                available_pairs.append((construction.lines[i], construction.lines[j]))
        # Circle to Circle
        for i in range(len(construction.circles)):
            for j in range(i, len(construction.circles)): # allow the same circle
                available_pairs.append((construction.circles[i], construction.circles[j]))
        # Line to Circle
        for line in construction.lines:
            for circle in construction.circles:
                available_pairs.append((line, circle))
        
        if len(available_pairs) == 0:
            return False
        
        self.states = random.choice(available_pairs)
        return True

    def apply(self):
        curve1, curve2 = self.states
        point1 = Point(self.construction.pop_point_label())
        point2 = Point(self.construction.pop_point_label())
        curve1.points.append(point1)
        curve2.points.append(point2)
        self.construction.add([point1, point2])
        line = Line(points=[point1, point2])
        self.construction.add(line)
        
        line_map = {}
        circ_map = {}
        if isinstance(curve1, Line):
            class1 = "line"
            line_map["curve1"] = curve1
        elif isinstance(curve1, Circle):
            class1 = "circle"
            circ_map["curve1"] = curve1
        else:
            raise ValueError("Unknown curve type")
        if isinstance(curve2, Line):
            class2 = "line"
            line_map["curve2"] = curve2
        elif isinstance(curve2, Circle):
            class2 = "circle"
            circ_map["curve2"] = curve2
        else:
            raise ValueError("Unknown curve type")
        
        return ConstructionDescription(
            description="Construct two points {A} and {B} on %s {curve1} and %s {curve2} respectively, and connect them with a segment."%(class1, class2),            
            point_map={"A": point1, "B": point2},
            line_map=line_map,
            circle_map=circ_map,
        )

class ConstructPointAndConnectExisting(ConstructionProcessor):
    def prepare(self):
        if len(self.construction.point_available_labels) < 1:
            return False
        available_pairs: List[Tuple[Point, Union[Line, Circle]]] = []
        for point in self.construction.points:
            lines = self.construction.point_on_lines(point)
            circles = self.construction.point_on_circles(point)
            for line in lines:
                if point in line.points:
                    continue
                available_pairs.append((point, line))
            for circle in circles:
                available_pairs.append((point, circle))
        if len(available_pairs) == 0:
            return False
        self.states = random.choice(available_pairs)
        return True

    def apply(self):
        point, curve = self.states
        new_point = Point(self.construction.pop_point_label())
        curve.points.append(new_point)
        self.construction.add(new_point)
        new_line = Line(points=[point, new_point])
        self.construction.add(new_line)
        curve_class = "line" if isinstance(curve, Line) else "circle"
        line_map = {"AB": new_line}
        circ_map = {}
        if isinstance(curve, Circle):
            circ_map["curve"] = curve
        else:
            line_map["curve"] = curve
        
        return ConstructionDescription(
            description="Construct a point {A} on %s {curve} and connect it to point {B}."%(curve_class),
            point_map={"A": new_point, "B": point},
            line_map=line_map,
            circle_map=circ_map,
        )

class ConstructionGenerator:
    def __init__(self, construction: GeometryConstruction):
        self.construction = construction
        self.available_operations: List[ConstructionProcessor] = [
            ConstructOrthocentre(construction),
            ConstructCircumcentre(construction),
            ConstructIncentre(construction),
            ConstructSegment(construction),
            ConstructTwoPointsAndConnect(construction),
            ConstructPointAndConnectExisting(construction),
        ]
        
    def create_triangle(self):
        point1 = Point(self.construction.pop_point_label())
        point2 = Point(self.construction.pop_point_label())
        point3 = Point(self.construction.pop_point_label())
        
        line1 = Line(points=[point1, point2])
        line2 = Line(points=[point2, point3])
        line3 = Line(points=[point3, point1])
        
        self.construction.add([point1, point2, point3, line1, line2, line3])
        return ConstructionDescription(
            description="Construct a triangle with vertices {A}, {B}, and {C}.",
            point_map={"A": point1, "B": point2, "C": point3},
        )
    
    def create_quadrilateral(self):
        point1 = Point(self.construction.pop_point_label())
        point2 = Point(self.construction.pop_point_label())
        point3 = Point(self.construction.pop_point_label())
        point4 = Point(self.construction.pop_point_label())
        
        line1 = Line(points=[point1, point2])
        line2 = Line(points=[point2, point3])
        line3 = Line(points=[point3, point4])
        line4 = Line(points=[point4, point1])
        
        self.construction.add([point1, point2, point3, point4, line1, line2, line3, line4])
        return ConstructionDescription(
            description="Construct a quadrilateral with vertices {A}, {B}, {C}, and {D}.",
            point_map={"A": point1, "B": point2, "C": point3, "D": point4},
        )
    
    def create_circle(self, with_center=True):
        if with_center:
            center = Point(self.construction.pop_point_label())
            circle = Circle(center=center)
            self.construction.add(circle)
            return ConstructionDescription(
                description="Construct a circle with center {O}.",
                point_map={"O": center},
            )
        else:
            circle = Circle()
            self.construction.add(circle)
            return ConstructionDescription(
                description="Construct a circle.",
            )
    
    def generate(self, extra_steps: int, return_str: bool = True, verbose: bool = False):
        r = random.random()
        ret = []
        if r < 0.5:
            ret.append(self.create_triangle())
        elif r < 0.8:
            ret.append(self.create_quadrilateral())
        else:
            ret.append(self.create_circle(random.random() < 0.7))
        
        if return_str:
            ret[-1] = ret[-1].format()
            
        for _ in range(extra_steps):
            construction = self.generate_step(verbose)
            if not construction:
                if verbose:
                    print("No more steps can be generated.")
                break
            ret.append(construction if not return_str else construction.format())
        return ret
        
    def generate_step(self, verbose: bool = False) -> Optional[ConstructionDescription]:
        available_operations = list(filter(lambda x: x.prepare(), self.available_operations))
        if len(available_operations) == 0:
            return None
        operation = random.choice(list(available_operations))
        if verbose:
            print(f"Performing operation: {operation.__class__.__name__}")
        description = operation.apply()
        if verbose:
            # print(f"Operation Applied: {description.description}")
            # print({**description.point_map, **description.line_map, **description.circle_map})
            print(f"Operation Applied: {description.format()}")
        return description


def _unit_test(step):
    shape = GeometryConstruction()
    generator = ConstructionGenerator(shape)
    generator.generate(step, verbose=True)
    print("Points:", [point.label for point in shape.points])
    print("Lines:", [[point.label for point in line.points] for line in shape.lines])
    print("Circles:", [[point.label for point in circle.points] for circle in shape.circles])
    print("Constraints:", [type(constraint).__name__ for constraint in shape.constraints])
    json_value = shape.json()
    print("JSON:")
    print(json.dumps(json_value))
    
    shape2 = GeometryConstruction.from_json(json_value)
    assert [point.label for point in shape.points] == [point.label for point in shape2.points]
    assert [[point.label for point in line.points] for line in shape.lines] == [[point.label for point in line.points] for line in shape2.lines]
    assert [[point.label for point in circle.points] for circle in shape.circles] == [[point.label for point in circle.points] for circle in shape2.circles]
    assert [type(constraint).__name__ for constraint in shape.constraints] == [type(constraint).__name__ for constraint in shape2.constraints]
    # print('\033[1;36m' + 'Unit Test Passed!' + '\033[0m')

def unit_test():
    for i in range(10):
        for j in range(3):
            _unit_test(i)
            print("=====================")
    print('\033[1;36m' + 'Unit Test Passed!' + '\033[0m')


if __name__ == '__main__':
    unit_test()