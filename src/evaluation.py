import os, json, sys, time, re, math, random, datetime, argparse, requests
from typing import *
from src.definition import *
import numpy as np
from scipy.optimize import linear_sum_assignment

class Matching:
    def __init__(self, score: np.ndarray):
        self.score = score
        self._calculate()

    def _calculate(self):
        m, n = self.score.shape
        
        if m == 0 or n == 0:
            self.precision = 1. if m + n == 0 else 0.
            self.recall = 1. if m + n == 0 else 0.
            self.row_ind = np.zeros((0,), dtype=int)
            self.col_ind = np.zeros((0,), dtype=int)
            self.total_score = 0.0
            self.f1 = 1. if m + n == 0 else 0.
            return

        cost = -self.score
        self.row_ind, self.col_ind = linear_sum_assignment(cost)
        
        total_score = 0.0
        for i, j in zip(self.row_ind, self.col_ind):
            if i < m and j < n:
                total_score += self.score[i, j].item()

        self.precision = total_score / m
        self.recall    = total_score / n
        self.total_score = total_score
        self.f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall) if self.precision + self.recall > 0 else 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }

class ConstructionMatching:
    def __init__(self, pred: GeometryConstruction, target: GeometryConstruction):
        self.pred = pred
        self.target = target
        self._calculate()
    
    def _calculate(self):
        self.points = Matching(score_matrix(self.pred.points, self.target.points))
        self.lines = Matching(score_matrix(self.pred.lines, self.target.lines))
        self.circles = Matching(score_matrix(self.pred.circles, self.target.circles))
        self.constraints = Matching(score_matrix(self.pred.constraints, self.target.constraints))
        self.scores = {
            "points": self.points.to_dict(),
            "lines": self.lines.to_dict(),
            "circles": self.circles.to_dict(),
            "constraints": self.constraints.to_dict(),
        }

def score_matrix(points_pred: List[GeometryElement], points_target: List[GeometryElement]):
    m = len(points_pred)
    n = len(points_target)
    scores = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            scores[i, j] = matching_score(points_pred[i], points_target[j])
    return scores

def matching_score(
    pred: GeometryElement,
    target: GeometryElement
) -> float:
    if isinstance(pred, Point) and isinstance(target, Point):
        return 1.0 if pred.label == target.label else 0.0
    elif isinstance(pred, Line) and isinstance(target, Line):
        matching = Matching(score_matrix(pred.points, target.points))
        return matching.f1
    elif isinstance(pred, Circle) and isinstance(target, Circle):
        center_score = matching_score(pred.center, target.center)
        matching = Matching(score_matrix(pred.points, target.points))
        return 0.5 * (matching.f1 + center_score)
    elif isinstance(pred, Constraint) and isinstance(target, Constraint):
        if type(pred) != type(target):
            return 0.0
        if isinstance(pred, ConstraintEqual) and isinstance(target, ConstraintEqual):
            score_matrix11 = score_matrix(pred.dist1, target.dist1)
            score_matrix12 = score_matrix(pred.dist1, target.dist2)
            score_matrix21 = score_matrix(pred.dist2, target.dist1)
            score_matrix22 = score_matrix(pred.dist2, target.dist2)
            matching1 = 0.5 * (Matching(score_matrix11).f1 + Matching(score_matrix22).f1)
            matching2 = 0.5 * (Matching(score_matrix12).f1 + Matching(score_matrix21).f1)
            return max(matching1, matching2)
        if isinstance(pred, ConstraintParallel) and isinstance(target, ConstraintParallel):
            scores = score_matrix([pred.line1, pred.line2], [target.line1, target.line2])
            return Matching(scores).f1
        if isinstance(pred, ConstraintPerpendicular) and isinstance(target, ConstraintPerpendicular):
            scores = score_matrix([pred.line1, pred.line2], [target.line1, target.line2])
            return Matching(scores).f1
        if isinstance(pred, ConstraintLineCircleTangent) and isinstance(target, ConstraintLineCircleTangent):
            return 0.5 * (matching_score(pred.line, target.line) + matching_score(pred.circle, target.circle))
        if isinstance(pred, ConstraintCircleCircleTangent) and isinstance(target, ConstraintCircleCircleTangent):
            scores = score_matrix([pred.circle1, pred.circle2], [target.circle1, target.circle2])
            return Matching(scores).f1
    return 0.0


