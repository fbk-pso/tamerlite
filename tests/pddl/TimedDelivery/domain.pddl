(define (domain timed-delivery)
  (:requirements :strips :typing :durative-actions :timed-initial-literals)
  (:types package location)
  (:predicates
    (at ?p - package ?l - location)
    (road ?from ?to - location)
    (open ?l - location)
    (free))
  (:durative-action carry
    :parameters (?p - package ?from ?to - location)
    :duration (= ?duration 3)
    :condition (and
      (at start (at ?p ?from))
      (at start (free))
      (over all (road ?from ?to))
      (at end (open ?to)))
    :effect (and
      (at start (not (at ?p ?from)))
      (at start (not (free)))
      (at end (at ?p ?to))
      (at end (free)))))
