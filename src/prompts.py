
ZEROSHOT_PROMPT_V0 = """
This is a formal language for describing geometric constructions and constraints in python format:

1. Constructions:

1.1 point(label: Optional[str] = None) -> Point

e.g. 

```python
A = point(label="A") # Creates a labeled point named "A".
M = point() # Creates an unlabeled existing point. If you need this point in the following part but it's not labeled in the figure, you can create it in this way. Describe how this point was constructed clearly in a comment, e.g. Intersection of the circle centered at O and line AB.
```

1.2 line(label: Option[str] = None, through: Optional[List[Point]] = None) -> Line

e.g.
```python
AB = line(through=[A, B, M]) # Please include all the points that on the line.
l1 = line(label="l", through=[O])
l2 = line(label="l", through=[])
```

1.3 circle(label: Optional[str] = None, center: Optional[Point] = None, through: Optional[List[Point]] = None) -> Circle

e.g.

```python
circle_O = circle(center=O, through=[A, P])
circle2 = circle(label="Γ", center=P, through=[])
gamma = circle(through=[A, B, C])
```

2. Constraints:

2.1 equal(segment1: Tuple[Point, Point], segment2: Tuple[Point, Point])

This constraint is used to set the lengths of two segments equal to each other.

2.2 perpendicular(line1: Line, line2: Line)

This constraint is used to set two lines perpendicular to each other.

2.3 parallel(line1: Line, line2: Line)
2.4 tangent(line: Line, circle: Circle)
2.5 assign_length(segment: Tuple[Point, Point], value: float)
2.6 assign_angle(vertex: Point, point1: Point, point2: Point, angle: float, radians: boolean = False)

3. Tags:

Tags are used to annotate geometric elements with additional information. They can be used to label segments, angles, or any other relevant information.

3.1 tag_segment(segment: Tuple[Point, Point], tag: str)
3.2 tag_angle(vertex: Point, point1: Point, point2: Point, tag: str)


Based on the provided image, please write out the geometric constructions and constraints using the specified format. Follow the guidelines below:

1. Code Organization:
Place all your code inside a single code block, arranged in the following order:
(1) constructions, (2) constraints, and (3) tags.

2. No Redefinitions:
All required classes and functions are already defined. Do not define any class or function. And do not use any undefined class or function.

3. No Imports or Built-ins:
Do not import any libraries or modules. Also, do not use any built-in functions.
"""


ZEROSHOT_PROMPT_V1 = """
This is a formal language for describing geometric constructions in python format:

1. Elements:

1.1 point(label: Optional[str] = None) -> Point

e.g. 

```python
A = point(label="A") # Creates a labeled point named "A".
M = point() # Creates an unlabeled existing point. If you need this point in the following part but it's not labeled in the figure, you can create it in this way. Describe how this point was constructed clearly in a comment, e.g. Intersection of the circle centered at O and line AB.
```

1.2 line(label: Option[str] = None, through: Optional[List[Point]] = None) -> Line

e.g.
```python
AB = line(through=[A, B, M]) # Please include all the points that on the line.
l1 = line(label="l", through=[O])
l2 = line(label="l", through=[])
```

1.3 circle(label: Optional[str] = None, center: Optional[Point] = None, through: Optional[List[Point]] = None) -> Circle

e.g.

```python
circle_O = circle(center=O, through=[A, P])
circle2 = circle(label="Γ", center=P, through=[])
gamma = circle(through=[A, B, C])
```

2. Constraints:

2.1 equal(segment1: Tuple[Point, Point], segment2: Tuple[Point, Point])
2.2 perpendicular(line1: Line, line2: Line)
2.3 parallel(line1: Line, line2: Line)
2.4 tangent_line_circle(line: Line, circle: Circle)
2.5 tangent_circle_circle(circle1: Circle, circle2: Circle)

Based on the provided image, please write out the geometric constructions using the specified format. Follow the guidelines below:

1. Code Organization:
Place all your code inside a single code block, arranged in the following order:
(1) elements, and (2) constraints.

2. No Redefinitions:
All required classes and functions are already defined. Do not define any class or function. And do not use any undefined class or function.

3. No Imports or Built-ins:
Do not import any libraries or modules. Also, do not use any built-in functions.
"""


CONSTRUCTION2NL_PROMPT = """{construction}

1. Polish the given statement directly without adding any additional text.
2. Use natural and professional language, incorporating appropriate mathematical notation and terminology.
3. Ensure clarity and simplicity by merging similar information and eliminating redundancy.
4. Refer to any unlabeled point as "an unlabeled point" in your text EXPLICITLY and assign it a letter name.
"""


CONSTRUCTION2NL_PROMPT_2 = """{construction}

Given the constructed diagram, describe it in a single, simple, and precise paragraph. Use natural and professional language with appropriate mathematical notation and terminology. The description should be clear and straightforward, avoiding unnecessary complexity while ensuring all key information is included. Merge related details to maintain conciseness. Explicitly identify any unlabeled point as "an unlabeled point" and assign it a letter name. Start with "The diagram".
"""


CAPTION_SAMPLE_PROMPT = """You are a geometric construction interpreter. Given the following diagram, describe its construction process in one concise and precise paragraph."""