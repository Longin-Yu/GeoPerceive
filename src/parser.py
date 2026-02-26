import ast
from src.definition import *

class ParserError(Exception):
    """Base class for all parser-related errors."""
    pass

class ParserInvalidSyntaxError(ParserError):
    """Raised when unsafe or disallowed syntax is detected."""
    pass

class ParserInvalidActionError(ParserError):
    """Raised when the code attempts to perform an illegal action."""
    pass

class ParserFailExecutionError(ParserError):
    """Raised when code execution fails."""
    pass

class Parser:
    def __init__(self):
        self._construction = GeometryConstruction()

        self._env = {
            "point": self._point,
            "line": self._line,
            "circle": self._circle,
            "equal": self._equal,
            "perpendicular": self._perpendicular,
            "parallel": self._parallel,
            "tangent_line_circle": self._tangent_line_circle,
            "tangent_circle_circle": self._tangent_circle_circle,
            # "assign_length": self._assign_length,
            # "assign_angle": self._assign_angle,
            # "tag_segment": self._tag_segment,
            # "tag_angle": self._tag_angle,
            "__builtins__": {}, 
        }

    # === 构造器 ===
    def _point(self, label=None):
        pt = Point(label)
        self._construction.points.append(pt)
        return pt

    def _line(self, label=None, through=None):
        # TODO: support label
        ln = Line(through)
        self._construction.lines.append(ln)
        return ln

    def _circle(self, label=None, center=None, through=None):
        # TODO: support label
        cir = Circle(center, through)
        self._construction.circles.append(cir)
        return cir

    def _equal(self, seg1, seg2):
        assert isinstance(seg1, tuple) and isinstance(seg2, tuple)
        assert len(seg1) == 2 and len(seg2) == 2
        assert isinstance(seg1[0], Point) and isinstance(seg1[1], Point)
        assert isinstance(seg2[0], Point) and isinstance(seg2[1], Point)
        self._construction.constraints.append(ConstraintEqual(seg1, seg2))

    def _perpendicular(self, l1, l2):
        assert isinstance(l1, Line) and isinstance(l2, Line)
        self._construction.constraints.append(ConstraintPerpendicular(l1, l2))

    def _parallel(self, l1, l2):
        assert isinstance(l1, Line) and isinstance(l2, Line)
        self._construction.constraints.append(ConstraintParallel(l1, l2))

    def _tangent_line_circle(self, line, circle):
        assert isinstance(line, Line) and isinstance(circle, Circle)
        self._construction.constraints.append(ConstraintLineCircleTangent(line, circle))

    def _tangent_circle_circle(self, circle1, circle2):
        assert isinstance(circle1, Circle) and isinstance(circle2, Circle)
        self._construction.constraints.append(ConstraintCircleCircleTangent(circle1, circle2))

    # def _assign_length(self, seg, value):
    #     self._construction.constraints.append(ConstraintAssignLength(seg, value))

    # def _assign_angle(self, v, p1, p2, angle, radians=False):
    #     self._construction.constraints.append(("assign_angle", v, p1, p2, angle, radians))

    # def _tag_segment(self, seg, tag):
    #     self._constructed["tags"].append(("tag_segment", seg, tag))

    # def _tag_angle(self, v, p1, p2, tag):
    #     self._constructed["tags"].append(("tag_angle", v, p1, p2, tag))

    def _check_ast_safe(self, code: str):
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as e:
            raise ParserInvalidSyntaxError(f"Syntax error: {e}")

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise ParserInvalidActionError("Import statements are not allowed.")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id not in self._env:
                    raise ParserInvalidActionError(f"Calling unknown function: {node.func.id}")
            if isinstance(node, ast.Attribute):
                raise ParserInvalidActionError("Attribute access is not allowed (e.g., obj.attr).")
            if isinstance(node, (ast.With, ast.Try, ast.Lambda, ast.FunctionDef, ast.ClassDef)):
                raise ParserInvalidActionError(f"Unsupported construct: {type(node).__name__}")
            if isinstance(node, ast.Global):
                raise ParserInvalidActionError("Global statements are not allowed.")

    def parse(self, code: str) -> GeometryConstruction:
        try:
            self._check_ast_safe(code)
            exec(code, self._env, {})
            return GeometryConstruction(
                points=self._construction.points,
                lines=self._construction.lines,
                circles=self._construction.circles,
                constraints=self._construction.constraints,
            )
        except ParserError:
            raise
        except Exception as e:
            raise ParserFailExecutionError(f"Execution failed: {e}")

    @property
    def points(self) -> List[Point]: return self._construction.points
    @property
    def lines(self) -> List[Line]: return self._construction.lines
    @property
    def circles(self) -> List[Circle]: return self._construction.circles
    @property
    def constraints(self) -> List[Constraint]: return self._construction.constraints
    # @property
    # def tags(self): return self._constructed["tags"]


def extract_code_block(text):
    pattern = r"```(?:[\w+-]*)\n(.*?)```"
    code_blocks = re.findall(pattern, text, re.DOTALL)
    if not code_blocks:
        return None
    else:
        return code_blocks[-1]
    
def dump_to_code(construction: GeometryConstruction) -> str:
    points = construction.points
    lines = construction.lines
    circles = construction.circles
    constraints = construction.constraints
    element2id: Dict[GeometryElement, str] = {}
    defs = []
    
    num_unlabeled_point = 0
    for point in points:
        id = point.label
        if id is None:
            num_unlabeled_point += 1
            id = f"unlabeled_point_{num_unlabeled_point}"
        suffix = 0
        current_id = id
        while current_id in element2id.values():
            suffix += 1
            current_id = f"{id}_{suffix}"
        element2id[point] = current_id
        defs.append(f"{current_id} = point(label=\"{current_id}\")")
        
    defs.append("")
    
    for i, line in enumerate(lines):
        id = f"line_{i+1}"
        element2id[line] = id
        defs.append(f"{id} = line(through=[{', '.join([element2id[point] for point in line.points])}])")
        
    defs.append("")
    
    for i, circle in enumerate(circles):
        id = f"circle_{i+1}"
        element2id[circle] = id
        if circle.center:
            defs.append(f"{id} = circle(center={element2id[circle.center]}, through=[{', '.join([element2id[point] for point in circle.points])}])")
        else:
            defs.append(f"{id} = circle(through=[{', '.join([element2id[point] for point in circle.points])}])")
        
    defs.append("")
    
    for constraint in constraints:
        if isinstance(constraint, ConstraintEqual):
            defs.append(f"equal(({element2id[constraint.dist1[0]]}, {element2id[constraint.dist1[1]]}), ({element2id[constraint.dist2[0]]}, {element2id[constraint.dist2[1]]}))")
        elif isinstance(constraint, ConstraintPerpendicular):
            defs.append(f"perpendicular({element2id[constraint.line1]}, {element2id[constraint.line2]})")
        elif isinstance(constraint, ConstraintParallel):
            defs.append(f"parallel({element2id[constraint.line1]}, {element2id[constraint.line2]})")
        elif isinstance(constraint, ConstraintLineCircleTangent):
            defs.append(f"tangent_line_circle({element2id[constraint.line]}, {element2id[constraint.circle]})")
        elif isinstance(constraint, ConstraintCircleCircleTangent):
            defs.append(f"tangent_circle_circle({element2id[constraint.circle1]}, {element2id[constraint.circle2]})")
        
    return "\n".join(defs)